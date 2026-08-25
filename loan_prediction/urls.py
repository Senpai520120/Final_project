from django.urls import path

from loan_prediction import views

app_name = "loan_prediction"

urlpatterns = [
    path("", views.home, name="home"),
]
