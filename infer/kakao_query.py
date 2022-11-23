from unittest import result
import requests, json, os, sys, time
import pymysql
import sqlalchemy as db
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from db_class import StoreClass, STORE_TABLE_NAME
from db_configure import *


location_return_url = ''
facilities_return_url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
json_file = 'temp.json'

Base = declarative_base()

result_dict = {
        'documents' : [],
        'meta' : {
            
        }
    }

def location_return(location):
    '''
    카카오 맵으로부터 도로명 주소를 입력 받으면 해당 주소의 위도 경도를 return 해주는 함수

    location(str) : 도로명 주소
    '''
    
    pass

def facilities_return(x, y, radius, keyword):
    '''
    위도, 경도, 반경을 입력받고 검색어를 입력받으면
    해당 위치 반경 내에 있는 검색어에 해당하는 시설의 정보를 dictionary 형태로 return 하는 함수

    x(float)     : 해당 위치의 위도
    y(float)     : 해당 위치의 경도
    radius(int)  : 해당 위치로 부터 반경
    keyword(str) : 해당 위치에서 찾을 시설
    '''
    global result_dict
    global json_file

    current_page = 1
    headers = {
        'Authorization': f'KakaoAK {os.environ["kakao"]}',
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    
    params = {
        'x': f'{x}',
        'y': f'{y}',
        'radius': f'{radius}',
        'page' : f'{current_page}'
    }

    # data
    data = f'query={keyword}'.encode('utf-8')

    # 응답 받아오기

    while True:
        try:
            response = requests.get(facilities_return_url, params=params, data=data, headers=headers).json()
            # result_dict의 documents에 response 의 documents 값을 append 해준다.
            #del response['distance']
            result_dict['documents'].extend(response['documents'])
            result_dict['meta'] = response['meta']
            # response is_end의 값이 True이면 return
            print(f'cafe length :  {len(result_dict["documents"])}')
            # 문자열로 계산된다 ;;
        
            if response['meta']['is_end']:
                break
            current_page += 1
            #print(current_page)
            params['page'] = str(current_page)
            time.sleep(0.5)
        except:
            break

    # json 으로 변환해서 파일에 저장
    # json을 파이썬 식(dictionary)으로 사용하려면 그대로 response return

    # json.dumps를 하면서 ascii code로 encoding 해서 문제가 생겼다. ensure_ascii option을 false 로 바꿔준다.
    #print(response)
    print(f'cafe length :  {len(result_dict["documents"])}')

    #print(json.dumps(response), file=f)

    return response

def read_result_dict():
    global result_dict
    try:
        with open(json_file, 'r') as temp:
            result_dict = json.load(temp)   
    except:
        result_dict = {
        'documents' : [],
        'meta' : {
            
        }
    }

def write_to_db(result_dict):
    #디비 연결
    engine = db.create_engine(f'mysql+pymysql://{user}:{passwd}@{host}:{port}/{database}')
    
    #이미 테이블 있으면 삭제한다.
    #engine.execute(f'DROP TABLE IF EXISTS {STORE_TABLE_NAME}')
    engine.execute(f'DELETE FROM {STORE_TABLE_NAME}')
    # 이 블럭은 table 을 만든다.
    
    #engine 에 종속적인 세션을 만든다.
    Session = sessionmaker(engine)
    session = Session() # 이거로 orm 통제
    
    
    for store_info in result_dict['documents']:
        #print(store_info['id'], store_info['place_name'], store_info['x'], store_info['y'])
        store = StoreClass(
                            store_info['id'], 
                            store_info['place_name'], 
                            store_info['road_address_name'], 
                            store_info['phone'], 
                            store_info['x'], 
                            store_info['y']
                        )
        session.add(store)
    session.commit()
  

facilities_return(126.936611826163, 37.55518625891015, 250, '카페')
facilities_return(126.936611826163, 37.55675399978744, 250, '카페')
facilities_return(126.936611826163, 37.55848393786034, 250, '카페')

facilities_return(126.934991512720, 37.55518625891015, 250, '카페')
facilities_return(126.934991512720, 37.55675399978744, 250, '카페')
facilities_return(126.934991512720, 37.55848393786034, 250, '카페')

facilities_return(126.939156399652, 37.55518625891015, 250, '카페')
facilities_return(126.939156399652, 37.55675399978744, 250, '카페')
facilities_return(126.939156399652, 37.55848393786034, 250, '카페')


# distance 통일시켜서 중복되게 유도한다.
for i in result_dict['documents']:
        i['distance'] = 0

# 중복 제거
result_dict['documents'] = [dict(t) for t in {tuple(d.items()) for d in result_dict['documents']}]

write_to_db(result_dict)

# 이제부터는 파일에서 불러와서 중복된거면 포함 안시키는 방향으로 바뀐다.