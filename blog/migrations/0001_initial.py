# Generated by Django 4.2.4 on 2023-08-04 05:21

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Car_info',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('handnos', models.CharField(max_length=20)),
                ('info_car_identify', models.CharField(max_length=1)),
                ('info_car_num', models.CharField(max_length=20)),
                ('info_disrepair_org_img', models.CharField(max_length=200)),
                ('info_disrepair_pre_scr', models.CharField(max_length=200)),
                ('info_disrepair_pre_sep', models.CharField(max_length=200)),
                ('info_disrepair_pre_cru', models.CharField(max_length=200)),
                ('info_disrepair_pre_bre', models.CharField(max_length=200)),
                ('info_disrepair_area_scr', models.IntegerField()),
                ('info_disrepair_area_sep', models.IntegerField()),
                ('info_disrepair_area_cru', models.IntegerField()),
                ('info_disrepair_area_bre', models.IntegerField()),
                ('info_disrepair_premon_scr', models.IntegerField()),
                ('info_disrepair_premon_sep', models.IntegerField()),
                ('info_disrepair_premon_cru', models.IntegerField()),
                ('info_disrepair_premon_bre', models.IntegerField()),
                ('info_total_pay', models.IntegerField()),
                ('info_disrepair_date', models.CharField(max_length=20)),
                ('info_estimate_date', models.CharField(max_length=20, null=True)),
                ('info_car_acc_num', models.CharField(max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('handnos', models.CharField(max_length=20)),
                ('user_names', models.CharField(max_length=20)),
                ('user_emails', models.CharField(max_length=20)),
                ('user_password', models.CharField(max_length=100)),
                ('user_date_created', models.DateTimeField()),
                ('user_birthdate', models.CharField(max_length=20)),
            ],
        ),
    ]
