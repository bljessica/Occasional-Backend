from django.db import models

# Create your models here.
class Picture(models.Model):
    '''图片'''
    image = models.ImageField(upload_to='static/images/')
    add_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.image)

    class Meta:
        verbose_name = '图片'
        verbose_name_plural = verbose_name

class Result(models.Model):
    name = models.CharField(max_length=1000)

    def __str__(self):
        return self.name
