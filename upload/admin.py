from django.contrib import admin
from .models import Picture

# Register your models here.
class PictureAdmin(admin.ModelAdmin):
    fields = ['image', 'add_time']
    list_display = ('image', 'add_time')
    search_fields = ['image']

admin.site.register(Picture, PictureAdmin)
