from django.urls import path
from. import views
urlpatterns = [
    path('',views.index,name='index'),
    path('about/',views.about,name="about"),
    path('dashboard/',views.dashboard,name="dashboard"),
    path('cargaExcel/',views.process_excel,name="cargaExcel"),
    path('confirmar_carga/', views.confirmar_carga, name='confirmar_carga'),


]