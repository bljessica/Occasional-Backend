from .models import Picture, Result
from rest_framework import serializers

class UploadSerializer(serializers.ModelSerializer):

    class Meta:
        model = Picture
        # fields = ('image', 'image_path')
        fields = '__all__'


class NameSerializer(serializers.ModelSerializer):
    name = serializers.CharField()

    class Meta:
        model = Result
        fields = '__all__'