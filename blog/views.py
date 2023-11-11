from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from . models import User, Car_info
from django.core.files.storage import FileSystemStorage
from django.db.models import Count
from django.contrib.messages import constants as messages_constants
from django.http import JsonResponse
from django.utils import timezone

# 라이브러리 불러오기
import os
import numpy as np
from PIL import Image
from collections import Counter
# from datetime import datetime
import torch
import cv2
import matplotlib.pyplot as plt

from src.Models import Unet

# 글로벌 변수
g_img = ''
g_img2 = ''
uploaded_file = ''


def login(request):
    return render(request, 'login.html', {})



# ###########################################################################################################################
# 함수 이미지 업로드 처리
# ###########################################################################################################################

def upload(request):
    global g_img, g_img2, uploaded_file  # uploaded_file 변수를 전역으로 선언



    if request.method == "GET":
        return HttpResponse('정상적인 경로가 아닙니다.')

    try:
        uploaded_file = request.FILES['file_input']  # 업로드된 파일 객체 저장
    except:
        return HttpResponse('업로드된 파일이 없습니다.')
    
    print('파일명 : ', uploaded_file.name)
    
    # 파일 저장
    up_image = uploaded_file.name
    
    # 이미지 서버에 저장
    fs = FileSystemStorage(location='media/upload/', base_url='media/upload')
    save_file = fs.save(up_image, uploaded_file)
    
    # 실제 저장된 파일 경로 확인
    upload_image_path = '/' + fs.url(save_file)
    print('실제 이미지 경로 : ', upload_image_path)
    
    # 글로벌 변수에 적용해주고 index로 이동
    g_img = '.' + upload_image_path
    g_img2 = upload_image_path
    
    return redirect('index')


def img_save(org_img, img_result, name_str, uploaded_file):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
    img = ax.imshow(org_img.astype('uint8'), alpha=0.5)
    img = ax.imshow(img_result, cmap='jet', alpha=0.5)
    ax.axis('off')
    fig.tight_layout()

    # 업로드된 파일의 이름 가져오기
    file_name = uploaded_file.name
    # 파일 경로 생성
    file_path = os.path.join('media/', name_str+ '/' + file_name)

    # 이미지 저장
    plt.savefig(file_path, dpi=300)

    # plt를 초기화
    plt.clf()
    
    return file_path

def search(request):
    if request.method == 'POST':
        car_number = request.POST.get('car_number')  # 차 번호
        phone_number = request.POST.get('phone_number')  # 휴대폰 번호

        # 데이터베이스에서 조회
        car_data = Car_info.objects.filter(info_car_num=car_number, handnos=phone_number).first()
        

        if car_data:
            return render(request, 'search_result.html', {'car_data': car_data})
        else:
            return render(request, 'search_result.html', {'not_found': True})
    else:
        return render(request, 'search_form.html')


def index(request):
    global g_img, g_img2
    img_data = []
    car_number = ''
    phone_number = ''
    
    print('g_img : ', g_img)
    if g_img:
        car_pay_list, scratch_area_list, img_path_list = ImageToCar(g_img, uploaded_file)
        save_to_db(car_pay_list, scratch_area_list, img_path_list)
        img_path = []        

        real_org_img = f'/media/upload/{uploaded_file}'
        for data in img_path_list:
            img_path.append(f'/{data}')
        
        img_data = [real_org_img, img_path[0], img_path[1], img_path[2], img_path[3]]
        print('이미지 경로 데이터 :', img_data)
    # for data in Car_info.objects.filter(handnos='').values()

    if request.method == 'POST':
        car_number = request.POST.get('car_number')  # 차 번호
        phone_number = request.POST.get('phone_number')  # 휴대폰 번호
    
    # 데이터베이스에서 조회
    log_data = search_db(car_number, phone_number)
    print('log_data : ', log_data)
    
    # 정보 출력( 가장 최근 것 가져오기 출력 
    latest_car_info = Car_info.objects.latest('id')    
    print('확인:', latest_car_info.info_disrepair_area_bre)
    if g_img:
        gg_img = g_img2
    else:
        gg_img = 'https://dummyimage.com/720x600'
    
    if not log_data:
        print("입력1")
        # print(len(g_img))
        print(latest_car_info)
        
        rtn_msg = {
            # 'logdata' : log_data,
            'log_data_1': latest_car_info if latest_car_info else None,  # 최근 차량 정보가 없을 때는 None 할당
            'g_img' : img_data,   
        }
        
    else:
        print("입력2")
        rtn_msg = {
            'logdata': log_data,
            'log_data_1': latest_car_info if latest_car_info else None,
            'g_img': gg_img,
        }    

    
    
    # rtn_msg = {
    #     'logdata' : log_data,
    #     # 'logimgdata' : log_img_data,
    #     'img_data_list' : img_data,   
    # }
    
    # 글로벌 변수 초기화  -> 새로고침하면 안나옴
    g_img = ''
    g_img2 = ''
    
    return render(request, 'index.html', rtn_msg)

def search_db(car_number, phone_number):
    car_data = Car_info.objects.filter(handnos=phone_number, info_car_num=car_number).order_by('-id')[:10]

    if car_data:
        result_list = []
        for car_info in car_data:
            result_list.append({
                'info_disrepair_premon_scr': car_info.info_disrepair_premon_scr,
                'info_disrepair_premon_sep': car_info.info_disrepair_premon_sep,
                'info_disrepair_premon_cru': car_info.info_disrepair_premon_cru,
                'info_disrepair_premon_bre': car_info.info_disrepair_premon_bre,
                'info_total_pay': car_info.info_total_pay,
                'info_disrepair_date': car_info.info_disrepair_date,
                'info_estimate_date': car_info.info_estimate_date,
            })
        print('car_)data 정보 : ',car_data)    
        
        return result_list
    else:
        return ''
    



# ###########################################################################################################################
# 인공지능 처리
# ###########################################################################################################################

# 인공지능 처리
def ImageToCar(imgPath, uploaded_file):
    # 학습된 모델 불러오기
    weight_path0 = './models/[DAMAGE][Scratch_0]Unet.pt'
    weight_path1 = './models/[DAMAGE][Seperated_1]Unet.pt'
    weight_path2 = './models/[DAMAGE][Crushed_2]Unet.pt'
    weight_path3 = './models/[DAMAGE][Breakage_3]Unet.pt'

    # 환경설정(정상,비정상클래스)
    n_classes = 2
    device = 'cpu'   # GPU가 있으시면 'cuda'

    # 모델0 불러오기
    model0 = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
    model0.model.load_state_dict(torch.load(weight_path0, map_location=torch.device(device)))
    model0.eval()

    # 모델1 불러오기
    model1 = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
    model1.model.load_state_dict(torch.load(weight_path1, map_location=torch.device(device)))
    model1.eval()

    # 모델2 불러오기
    model2 = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
    model2.model.load_state_dict(torch.load(weight_path2, map_location=torch.device(device)))
    model2.eval()

    # 모델3 불러오기
    model3 = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
    model3.model.load_state_dict(torch.load(weight_path3, map_location=torch.device(device)))
    model3.eval()
    
    org_img = cv2.imread(imgPath)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    org_img = cv2.resize(org_img, (256, 256))
    # print(org_img.shape)
    #이미지 전처리/정규화
    img_process = org_img / 255.
    # 전치행렬 처리
    img_process = img_process.transpose([2, 0, 1])
    # 인공지능이 처리할 수 있는 Tensor구조로 변경
    # 이유 : pyTorch는 처리되는 데이터는 모두 Tensor 구조로 변경
    img_process = torch.tensor(img_process).float().to(device)
    # 3차원 배열을 4차원으로 확장
    img_process = img_process.unsqueeze(0)
    results0 = model0(img_process)
    results1 = model1(img_process)
    results2 = model2(img_process)
    results3 = model3(img_process)
    # 예측결과에서 가장 높은 성능의 결과값만 뽑아냄
    img_results0 = torch.argmax(results0, dim=1).detach().cpu().numpy()
    img_results1 = torch.argmax(results1, dim=1).detach().cpu().numpy()
    img_results2 = torch.argmax(results2, dim=1).detach().cpu().numpy()
    img_results3 = torch.argmax(results3, dim=1).detach().cpu().numpy()
    # print(img_results.shape)
    # print(type(img_results))

    # 결과파일 (1, 256, 256)을 우리가 눈으로 확인할 수 있는 (256, 256, 1) 로 변경
    img_results0 = img_results0.transpose([1, 2, 0])
    img_results1 = img_results1.transpose([1, 2, 0])
    img_results2 = img_results2.transpose([1, 2, 0])
    img_results3 = img_results3.transpose([1, 2, 0])


    scr_path = img_save( org_img, img_results0, 'scratch_0', uploaded_file)
    sep_path = img_save( org_img, img_results1, 'seperated_1', uploaded_file)
    cru_path = img_save( org_img, img_results2, 'crushed_2', uploaded_file)
    bre_path = img_save( org_img, img_results3, 'breakage_3', uploaded_file)

    

    # 계산공식
    car_patch_price_0 = 30
    car_patch_price_1 = 50
    car_patch_price_2 = 40
    car_patch_price_3 = 80

    # 파손 부위 면적
    scratch_area0 = img_results0.sum()
    scratch_area1 = img_results1.sum()
    scratch_area2 = img_results2.sum()
    scratch_area3 = img_results3.sum()

    car_pay0 = format(scratch_area0 * car_patch_price_0, ',')
    car_pay1 = format(scratch_area1 * car_patch_price_0, ',')
    car_pay2 = format(scratch_area2 * car_patch_price_0, ',')
    car_pay3 = format(scratch_area3 * car_patch_price_0, ',')

    car_pay_list = [car_pay0, car_pay1, car_pay2, car_pay3]
    scratch_area_list = [scratch_area0, scratch_area1, scratch_area2, scratch_area3]
    img_path_list = [scr_path, sep_path, cru_path, bre_path]
    
    return  car_pay_list, scratch_area_list, img_path_list

# ###########################################################################################################################
# db처리
# ###########################################################################################################################

def save_to_db( car_pay_list, scratch_area_list, img_path_list):
    
    handnos_compulsion = '010-1111-1111'
    carnum_compulsion = '123가1234'
    current_time = timezone.now()
    count_car_acc = Car_info.objects.aggregate(total_count = Count('info_car_num'))
    
    car_identify = 0
    car_pay_int = []
    img_path = [] 
    
    org_path = g_img[1:]
    data = None
    
    for urldata in img_path_list:
        img_path.append(f'/{urldata}') 

    
    for pay in car_pay_list:
        pay_without_comma = pay.replace(',', '')  # 쉼표 제거
        car_pay_int.append(int(pay_without_comma))  # 정수로 변환

    print("car_pay_list:", car_pay_list)
    print("car_pay_int:", car_pay_int)
        
    if car_identify == 0:
        car_pay_sum0 = sum(car_pay_int)
    elif car_identify == 1:
        car_pay_sum0 = sum(car_pay_int) * 3
        
    if not data:
        return ''
    else:    
        data = Car_info(
            handnos = handnos_compulsion,
            info_car_identify = car_identify,
            info_car_num= carnum_compulsion,
            info_disrepair_org_img = org_path,
            info_disrepair_pre_scr = img_path[0],
            info_disrepair_pre_sep = img_path[1],
            info_disrepair_pre_cru = img_path[2],
            info_disrepair_pre_bre = img_path[3],
            info_disrepair_area_scr = scratch_area_list[0],
            info_disrepair_area_sep = scratch_area_list[1],
            info_disrepair_area_cru = scratch_area_list[2],
            info_disrepair_area_bre = scratch_area_list[3],
            info_disrepair_premon_scr = car_pay_int[0],
            info_disrepair_premon_sep = car_pay_int[1],
            info_disrepair_premon_cru = car_pay_int[2],
            info_disrepair_premon_bre = car_pay_int[3],
            info_total_pay = car_pay_sum0,
            info_disrepair_date = current_time.strftime('%Y-%m-%d/%H:%M'),
            info_estimate_date =  current_time.strftime('%Y-%m-%d/%H:%M'),
            info_car_acc_num = count_car_acc
        )
        data.save()
    







# def SearchCarinfo():
#     data = Car_info.objects.filter(info_phone=fid).first()
    
#     print(data.info_car_acc_num)
#     print(data.info_car_identify)

#     if not data:
#         return ''
#     else:
#         # 데이터가 존재하는 경우
        
        
#         return data







# # 학습된 모델 불러오기
# weight_path0 = 'models/[DAMAGE][Scratch_0]Unet.pt'
# weight_path1 = 'models/[DAMAGE][Seperated_1]Unet.pt'
# weight_path2 = 'models/[DAMAGE][Crushed_2]Unet.pt'
# weight_path3 = 'models/[DAMAGE][Breakage_3]Unet.pt'

# # 환경설정(정상,비정상클래스)
# n_classes = 2
# device = 'cpu'   # GPU가 있으시면 'cuda'

# # 모델0 불러오기
# model0 = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
# model0.model.load_state_dict(torch.load(weight_path0, map_location=torch.device(device)))
# model0.eval()

# # 모델1 불러오기
# model1 = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
# model1.model.load_state_dict(torch.load(weight_path1, map_location=torch.device(device)))
# model1.eval()

# # 모델2 불러오기
# model2 = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
# model2.model.load_state_dict(torch.load(weight_path2, map_location=torch.device(device)))
# model2.eval()

# # 모델3 불러오기
# model3 = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
# model3.model.load_state_dict(torch.load(weight_path3, map_location=torch.device(device)))
# model3.eval()




# print('모델 불러오기에 성공하였습니다.')

# def index(request):
#     return HttpResponse("연결")

# #분석할 이미지 불러오기 
# def read_ana_img(img):
#     org_img = cv2.imread(img)
#     org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
#     org_img = cv2.resize(org_img, (256, 256))
#     # print(org_img.shape)
#     #이미지 전처리/정규화
#     img_process = img / 255.
#     # 전치행렬 처리
#     img_process = img_process.transpose([2, 0, 1])
#     # 인공지능이 처리할 수 있는 Tensor구조로 변경
#     # 이유 : pyTorch는 처리되는 데이터는 모두 Tensor 구조로 변경
#     img_process = torch.tensor(img_process).float().to(device)
#     # 3차원 배열을 4차원으로 확장
#     img_process = img_process.unsqueeze(0)
#     results0 = model0(img_process)
#     results1 = model1(img_process)
#     results2 = model2(img_process)
#     results3 = model3(img_process)
#     # 예측결과에서 가장 높은 성능의 결과값만 뽑아냄
#     img_results0 = torch.argmax(results0, dim=1).detach().cpu().numpy()
#     img_results1 = torch.argmax(results1, dim=1).detach().cpu().numpy()
#     img_results2 = torch.argmax(results2, dim=1).detach().cpu().numpy()
#     img_results3 = torch.argmax(results3, dim=1).detach().cpu().numpy()
#     # print(img_results.shape)
#     # print(type(img_results))

#     # 결과파일 (1, 256, 256)을 우리가 눈으로 확인할 수 있는 (256, 256, 1) 로 변경
#     img_results0 = img_results0.transpose([1, 2, 0])
#     img_results1 = img_results1.transpose([1, 2, 0])
#     img_results2 = img_results2.transpose([1, 2, 0])
#     img_results3 = img_results3.transpose([1, 2, 0])

#     # 계산공식
#     car_patch_price_0 = 50

#     # 파손 부위 면적
#     scratch_area0 = img_results0.sum()
#     scratch_area1 = img_results1.sum()
#     scratch_area2 = img_results2.sum()
#     scratch_area3 = img_results3.sum()


#     car_pay0 = format(scratch_area0 * car_patch_price_0, ',')
#     car_pay1 = format(scratch_area1 * car_patch_price_0, ',')
#     car_pay2 = format(scratch_area2 * car_patch_price_0, ',')
#     car_pay3 = format(scratch_area3 * car_patch_price_0, ',')

