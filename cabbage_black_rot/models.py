from django.db import models

class PredictionHistory(models.Model):
    image = models.ImageField(upload_to='predictions/')
    prediction_class = models.CharField(max_length=100)
    confidence = models.FloatField()
    insight = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prediction_class} ({self.confidence:.2f})"
