import requests
import sqlalchemy as db
import infer

from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from db_class import StoreClass, CommentClass, STORE_TABLE_NAME, COMMENT_TABLE_NAME
from db_configure import *


# 가성비 effective
# 청결 clean
# 맛 tasty 
# 분위기 vibe
# 친절 kind
LABEL_COLUMNS = ['effective', 'clean', 'tasty', 'vibe', 'kind']

class LenError(Exception):
    def __str__(self):
        return "length does not match"

def get_comments(response):
    '''
    api 가 들어있는 url 이다.
    만약에 response['comment']['hasNext'] 값이 False라면 
    response['comment']를 return 아니라면 hasNext값이 False가 될 때까지 loop를 돌면서 
    https://place.map.kakao.com/commentlist/v/(가게 id)/(마지막 comment 의 comment_id)
    '''
    #print(f"response has next {response['comment']['hasNext']}")

    # 만약에 hasNext가 false라면 원래 값을 return 해준다.

    #while loop 을 돌면서 response['commnet']['list']에 값을 추가하는 방식
    while response['comment']['hasNext']:
        # 항상 2번 key값이 마지막이다.
        # 맨 마지막 key 값을 return 하게 한다.
        #https://stackoverflow.com/questions/16125229/last-key-in-python-dictionary
        last_comment = response['comment']['list'][-1]
        
        last_comment_id = last_comment['commentid']
        store_id = response['basicInfo']['cid']
        
        #print(f'last comment id is : {last_comment_id} store id is : {store_id}')
        
        # 새로운 comment를 받아줄 url이다.
        comment_retrive_url = f'https://place.map.kakao.com/commentlist/v/{store_id}/{last_comment_id}'
        
        comment_response = requests.get(comment_retrive_url).json()
        
        # 여기서 나온 댓글을 새로 목록에 추가한다.
        #print(comment_response)
        
        # expend 해주기
        response['comment']['list'].extend(comment_response['comment']['list'])
        # hasNext를 comment_response 의 hasNext로 바꿔준다.
        response['comment']['hasNext'] = comment_response['comment']['hasNext']        

def comment_db_write(comment_list, store_id):
    # 키워드 dictionary 미리 선언을 한다.
    # 이후에 comment_list 를 순회하면서 장점을 전부 더해서 store 테이블에 넣어준다.
    # 키워드 dictionary {} 값은 모두 0으로 초기화
    total_good_list = [0, 0, 0, 0, 0] # 댓글에서 나오는 장점의 총합

    for comment in comment_list:
        # datetime 형식으로 바꾸는게 낫겠지?        
        # 문자열 없거나 길이 초과시 대처법
        # 문자열 없으면 빈 문자열로 만들고 아니면 글자 개수 제한으로 절제
        comment_temp = comment.get('contents')
        comment_contents = "" if not comment_temp else comment_temp[:512] 
        # hard coding 말고 변수를 추가하자 그래야 db 랑 일관성 유지가 가능
        
        #keyword = [for i in list()]

        #photolist 받아오기
        photo_list_string = ""
        try:
            if comment.get('photoCnt'):
                photo_list_string = str([x['url'] for x in comment.get('photoList')[0:2]])
        except:
            if comment.get('photoCnt'):
                print("에러")
            pass
        #print(str(photo_list))
        
        
        #장점 classification 해주기
        #good_side = infer()

        good_list = infer.judge(comment)
        #dictionary에 각각 더해준다.
        
        for idx in range(0, len(good_list)):
            total_good_list[idx] += good_list[idx]
        
        # 또한 photoCnt 를 통해서 0이 아니라면 사진을 받아오는 방식을 사용한다.
        # 사진은 문자열로 받을지 아니면 링크로 받을지는 아직 미정
        
        c = CommentClass(
                             comment['commentid'],  
                             comment_contents,
                             comment.get('point'),
                             comment.get('photoCnt'), 
                             comment.get('likeCnt'),
                             comment.get('kakaoMapUserId'),
                             comment.get('username'),
                             photoList=photo_list_string,
                             strengths="",
                             userCommentCount=comment['userCommentCount'],
                             userCommentAverageScore=comment['userCommentAverageScore'],
                             date=comment['date'],
                             store_id=store_id,
                             effective=good_list[0],
                             clean=good_list[1],
                             tasty=good_list[2],
                             vibe=good_list[3],
                             kind=good_list[4]
                            )
        session.add(c)
    
    session\
    .query(StoreClass)\
    .filter_by(id=store_id)\
    .update({
            'effective' : total_good_list[0], 
            'clean' : total_good_list[1],
            'tasty' : total_good_list[2],
            'vibe' : total_good_list[3],
            'kind' : total_good_list[4],
            })

def comment_score_write(score_sum, score_count, store_id):
    #comment scoresum scorecount
    try :
        session\
        .query(StoreClass)\
        .filter_by(id=store_id)\
        .update({'star_mean' : round(int(score_sum)/int(score_count), 1), 'comment_count' : score_count})
        # 여기서 score count를 update를 해줄 것!
        # {'star_mean' : round(int(score_sum)/int(score_count), 1), 'comment_count' : 'score_count'}
    except:
        print("no star!\n")
        return

def one_store_analyze(store_id):
    
    info_url = f'https://place.map.kakao.com/main/v/{store_id}'
    
    # basicInfo의 feedback은 댓글나열한거
    # s2 graph가 매장의 정보를 나열한거
    print(info_url)
    response = requests.get(info_url).json()

    # 만약에 comment라는 key 값이 없으면 그냥 return
    
    # comment 보기 전에 mainphotourl 은 찾고 간다.
    # 없으면 그냥 냅두고~
    # 일단 varchar(512로 간다.)
    #print(response)
    mainphotourl=""
    # 삼항 연산자로 바꾼다. 아니면 try catch로 바꾸면 ㄱㅊ은가? 현재는 디버깅용
    try:
        mainphotourl= response.get('basicInfo').get('mainphotourl')
    except:
        print('메인 사진 없다~')
    #print(f'main mainphotourl : {mainphotourl}')
    
    # update 하기
    session.query(StoreClass).filter_by(id=store_id).update({'main_photo' : mainphotourl})

    if 'comment' not in response.keys() or 'list' not in response['comment'].keys():
        print('리뷰 엥꼬~')
        return

    # get_commnet 에서 댓글 가져오기 json 형식
    get_comments(response)

    #comment scoresum scorecount  해서 평균을 구한다.
    #print(response.get('comment'))
    comment_score_write(response.get('comment').get('scoresum'), response.get('comment').get('scorecnt'), store_id)
    comment_db_write(response['comment']['list'], store_id)

    # basicinfo mainphotorul
    # 카페 chart 안에서 update 하는 식으로 한다. 
    # 이거 update query 보내면 될꺼 같음
    
    ## comment 디비에 저장

    #print(response['comment']['list'])
    #print(len(response['comment']['list']))

def read_store_from_database():
    
    stmt = db.select(StoreClass)
    # store 의 id 모음
    ret = [x.id for x in session.scalars(stmt)]
    
    return ret

def comment_db_control():
    engine.execute(f'DELETE FROM {COMMENT_TABLE_NAME}')

if __name__ == '__main__':
    
    
    # 여기에 모델 불러와서 infer를 하는 방식이 가능할꺼 같음
    
    
    
    # model.어쩌구
    # 그리고 comment 가져오는 함수 안에서 infer 하기
    
    engine = db.create_engine(f'mysql+pymysql://{user}:{passwd}@{host}:{port}/{database}')
    Session = sessionmaker(engine)
    session = Session() # 이거로 orm 통제
    

    store_id_list = read_store_from_database()
    #print(store_id_list)
    #exit()
    
    comment_db_control()
    
    #one_store_analyze(1011256721)
    #session.commit()
    #comment.get('contents'),
    for id in store_id_list:
        one_store_analyze(id)
        session.commit()
    
    #one_store_analyze(doc)
    
