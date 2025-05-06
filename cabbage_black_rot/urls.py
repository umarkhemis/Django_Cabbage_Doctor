
from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    # path('register/', views.RegisterView.as_view(), name='register'),
    path('register/', views.register, name='register'),
    path('token/', views.CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('predict/', views.predict),
    path('history/', views.get_history),
    # path('ask/', views.ask_bot),
]