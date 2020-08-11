from django.urls import path, include
from .views import UploadPictureAPIView, GetResult

urlpatterns = [
    path(r'upload/', UploadPictureAPIView.as_view()),
    path(r'getName/', GetResult.as_view()),
]