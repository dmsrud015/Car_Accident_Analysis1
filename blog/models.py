from django.db import models

class  User(models.Model):
    id = models.AutoField(primary_key=True)
    handnos = models.CharField(max_length=20)
    user_names = models.CharField(max_length=20)
    user_emails = models.CharField(max_length=20)
    user_password = models.CharField(max_length=100)
    user_date_created = models.DateTimeField()
    user_birthdate = models.CharField(max_length=20)
    # user_Field = models.CharField(max_length=255)


def __str__(self):
    return self.handnos


class Car_info(models.Model):
    id = models.AutoField(primary_key=True)
    handnos = models.CharField(max_length=20)                               # 휴대폰
    info_car_identify = models.CharField(max_length=1)                      # 국산 or 수입
    info_car_num = models.CharField(max_length=20)                          # 차넘버
    
    info_disrepair_org_img = models.CharField(max_length=200)               # 원본 이미지 주소    
    info_disrepair_pre_scr = models.CharField(max_length=200)               # 스크래치
    info_disrepair_pre_sep = models.CharField(max_length=200)               # 이격
    info_disrepair_pre_cru = models.CharField(max_length=200)               # 찌그러짐
    info_disrepair_pre_bre = models.CharField(max_length=200)               # 파손
    
    info_disrepair_area_scr = models.IntegerField()                         # 스크래치 범위
    info_disrepair_area_sep = models.IntegerField()                         # 이격
    info_disrepair_area_cru = models.IntegerField()                         # 찌그러짐
    info_disrepair_area_bre = models.IntegerField()                         # 파손
    
    info_disrepair_premon_scr = models.IntegerField()                       # 스크래치 금액
    info_disrepair_premon_sep = models.IntegerField()                       # 이격
    info_disrepair_premon_cru = models.IntegerField()                       # 찌그러짐
    info_disrepair_premon_bre = models.IntegerField()                       # 파손
    info_total_pay = models.IntegerField()                                  # 총 예상 견적
    
    info_disrepair_date = models.CharField(max_length=20)                   # 파손날짜
    info_estimate_date = models.CharField(max_length=20, null=True)         # 견적날짜
    info_car_acc_num = models.CharField(max_length=20)                      # 사고 횟수

    
def __str__(self):
    return self.info_car_num