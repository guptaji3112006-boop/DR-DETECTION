from django.urls import re_path, include
from django.contrib import admin
from app.views import index

urlpatterns = [
    re_path(r'^$', index),
    re_path(r'^admin/', admin.site.urls),
]
