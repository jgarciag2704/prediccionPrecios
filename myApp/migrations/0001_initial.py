# Generated by Django 5.1.6 on 2025-02-05 15:52

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='historicoPrecios',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Fecha', models.DateField()),
                ('Nombre', models.TextField()),
                ('Calidad', models.TextField()),
                ('Presentacion', models.TextField()),
                ('Origen', models.TextField()),
                ('mercadoDeAbastos', models.TextField()),
                ('precioMinimo', models.DecimalField(decimal_places=2, max_digits=7, validators=[django.core.validators.MinValueValidator(0)])),
                ('precioMaximo', models.DecimalField(decimal_places=2, max_digits=7, validators=[django.core.validators.MinValueValidator(0)])),
                ('preciopromedio', models.DecimalField(decimal_places=2, max_digits=7, validators=[django.core.validators.MinValueValidator(0)])),
            ],
        ),
    ]
