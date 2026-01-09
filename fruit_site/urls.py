from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from detector.views import predict_image

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', predict_image, name='home'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)