from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import generics, permissions, status
from rest_framework.permissions import AllowAny
from .serializers import RegisterSerializer, CustomTokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from django.core.files.base import ContentFile
import google.generativeai as genai
from django.conf import settings
from .models import PredictionHistory
from .serializers import PredictionHistorySerializer
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from django.contrib.auth.models import User
# import openai
# from openai import OpenAI
import os
# from .genai_insights import get_disease_insight

genai.configure(api_key='AIzaSyB9IHWHbqggP__-hN9304vrJqTnvTDha3c')


# Load your trained model
Model_Path = os.path.join(os.path.dirname(__file__), 'Models', 'fine_tuned_cabbage_black_rot_detector_model.tflite')
# model = load_model(Model_Path)
# model = tf.keras.load_model(Model_Path)
# model = tf.keras.models.load_model('Models/fine_tuned_cabbage_black_rot_detector_model.tflite')

interpreter = tf.lite.Interpreter(model_path=Model_Path)
# interpreter = tf.lite.Interpreter(model_path="./Models/fine_tuned_cabbage_black_rot_detector_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


class_names = ['Cabage_black_rot', 'healthy', 'not_related_to_cabbage_black_rot']



@api_view(['POST'])
def predict(request):
    try:
        base64_image = request.data.get('image')  # Expecting JSON payload with 'image': <base64string>

        if not base64_image:
            return Response({'error': 'No image provided'}, status=400)

        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        image_file = io.BytesIO(image_data)
        image = Image.open(image_file).resize((160, 160)).convert('RGB')

        # Prepare for prediction
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        index = np.argmax(output_data)
        prediction_class = class_names[index]
        confidence = float(output_data[index])
        insight = generate_insight(prediction_class)

        # Save image to DB (optional)
        image_content = ContentFile(image_data, name='uploaded.jpg')
        entry = PredictionHistory.objects.create(
            image=image_content,
            prediction_class=prediction_class,
            confidence=confidence,
            insight=insight
        )

        return Response({
            'prediction': prediction_class,
            'confidence': confidence,
            'insight': insight,
            'image_url': request.build_absolute_uri(entry.image.url)
        })

    except Exception as e:
        raise ValidationError({"error": str(e)})




@api_view(['GET'])
def get_history(request):
    queryset = PredictionHistory.objects.order_by('-timestamp')
    serializer = PredictionHistorySerializer(queryset, many=True, context={'request': request})
    return Response(serializer.data)



@api_view(['POST'])
def ask_bot(request):
    question = request.data.get('question')
    if not question:
        return Response({'reply': 'No question received.'}, status=400)

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(question)

        # Safely extract the text
        if hasattr(response, 'text'):
            answer = response.text.strip()
        else:
            answer = 'No text reply from Gemini.'

        return Response({'reply': answer})

    except Exception as e:
        print(f"Error in ask_bot: {e}")  # See this in logs
        return Response({'reply': f'Error: {str(e)}'}, status=500)











# @api_view(['POST'])
# def ask_bot(request):
#     question = request.data.get('question')
#     if not question:
#         return Response({'reply': 'No question received.'}, status=400)

#     try:
#         # Use Gemini Pro (text-only)
#         model = genai.GenerativeModel('gemini-pro')
#         response = model.generate_content(question)

#         answer = response.text.strip() if hasattr(response, 'text') else 'No response from Gemini.'
#         return Response({'reply': answer})

#     except Exception as e:
#         return Response({'reply': f'Error: {str(e)}'}, status=500)





# @api_view(['POST'])
# def ask_bot(request):
#     question = request.data.get('question')
#     if not question:
#         return Response({'reply': 'No question received.'}, status=400)

#     # Use OpenAI or any other chatbot model
    
#     client = OpenAI()

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "user", "content": "question"}
#         ]
#     )

#     print(response.choices[0].message.content)
    
    
    # response = openai.ChatCompletion.create(
    #     model='gpt-3.5-turbo',
    #     messages=[{"role": "user", "content": question}],
    #     max_tokens=150
    # )
    # reply = response['choices'][0]['message']['content']
    # return Response({'reply': reply})


def generate_insight(pred_class):
    # Simple hardcoded insight; you can improve this
    insights = {
        "black_rot": "Black rot is caused by Xanthomonas campestris. Control it with crop rotation and resistant varieties.",
        "healthy": "Your cabbage appears healthy. Keep monitoring regularly.",
        "not_related": "This image doesn't appear to be a cabbage or related disease."
    }
    return insights.get(pred_class, "No insight available.")



# class RegisterView(generics.CreateAPIView):
#     serializer_class = RegisterSerializer
#     permission_classes = [AllowAny]

@api_view(['POST'])
def register(request):
    username = request.data.get('username')
    email = request.data.get('email')
    password = request.data.get('password')

    if not username or not email or not password:
        return Response({'error': 'All fields are required'}, status=400)

    if User.objects.filter(username=username).exists():
        return Response({'error': 'Username already exists'}, status=400)

    user = User.objects.create_user(username=username, email=email, password=password)
    return Response({'message': 'User registered successfully'})


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer