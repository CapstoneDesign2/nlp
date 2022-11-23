import sqlalchemy as db
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

STORE_TABLE_NAME = 'test_Cafe'
COMMENT_TABLE_NAME = 'test_Review'

Base = declarative_base()

# todo 팀원들이 원하는 field 값 추가해서 update 하기!
# 도로명 주소 string 일단 되는거 같고
# 대표 이미지 string 
# 별점 평균 float
class StoreClass(Base):
    __tablename__ = STORE_TABLE_NAME  # 데이터베이스에서 사용할 테이블 이름입니다.

    id = db.Column(db.Integer, primary_key=True)
    place_name = db.Column(db.String(50))
    road_address_name = db.Column(db.String(100))
    phone = db.Column(db.String(30))
    x = db.Column(db.Float)
    y = db.Column(db.Float)

    main_photo = db.Column(db.String(100))
    star_mean = db.Column(db.Float)
    
    comment_count = db.Column(db.Integer)
    bookmark_cnt = db.Column(db.Integer)

    effective = db.Column(db.Integer)
    clean = db.Column(db.Integer)
    tasty = db.Column(db.Integer)
    vibe = db.Column(db.Integer)
    kind = db.Column(db.Integer)

    # relationship 을 해야하나?
    #['effective', 'clean', 'tasty', 'vibe', 'kind']
    #addresses = relationship("Address", back_populates="user") // 다른 테이블과 foreign key 관계일 때 사용하는거 같음
    # init 할 때는 사진이랑 별점 평균 update 하지 말기! 그냥 null 값으로 초기화만하고 나중에 parser 에서 update 하기
    def __init__(self, id, place_name, road_address_name, phone, x, y):
        self.id = id
        self.place_name = place_name
        self.road_address_name = road_address_name
        self.phone = phone
        self.x = x
        self.y = y

        self.main_photo = ""
        self.star_mean = 0
        self.comment_count = 0
        self.bookmark_cnt = 0

        # 장점 개수
        self.effective = 0
        self.clean = 0
        self.tasty = 0
        self.vibe = 0
        self.kind = 0

    
    def __repr__(self):
       return f"User(id={self.id!r}, name={self.place_name!r}, phone={self.phone!r}, x={self.x}, y={self.y})"

class CommentClass(Base):
    __tablename__ = COMMENT_TABLE_NAME  # 데이터베이스에서 사용할 테이블 이름입니다.

    id = db.Column(db.Integer, primary_key=True)
    contents = db.Column(db.String(100))
    point = db.Column(db.Integer)
    photoCnt = db.Column(db.Integer)
    likeCnt = db.Column(db.Integer)
    kakaoMapUserId = db.Column(db.String(100))
    username = db.Column(db.String(30))
    photoList = db.Column(db.String(1024))
    # strengths
    userCommentCount = db.Column(db.Integer)
    userCommentAverageScore = db.Column(db.Float) # 정확도 크게 상관 없음
    date = db.Column(db.String(10)) # 이거 맞나? . . 으로 분리되어서 date 를 나타냄
    #keyword = 
    #['effective', 'clean', 'tasty', 'vibe', 'kind']
    store_id = db.Column(db.Integer, db.ForeignKey(f'{STORE_TABLE_NAME}.id', ondelete='CASCADE'))
    
    effective = db.Column(db.Integer)
    clean = db.Column(db.Integer)
    tasty = db.Column(db.Integer)
    vibe = db.Column(db.Integer)
    kind = db.Column(db.Integer)


    #addresses = relationship("Address", back_populates="user") // 아직은 코드상에서 댓글로 사용자 찾는 경우는 없는 것 같으니 사용하지 않는다.
    def __init__(self,    
                 id,
                 contents,
                 point, 
                 photoCnt, 
                 likeCnt, 
                 kakaoMapUserId,
                 username, 
                 photoList="", 
                 strengths="", 
                 userCommentCount=0, 
                 
                 userCommentAverageScore=0.0, 
                 date="",
                 store_id=0,
                 effective=0,
                 clean=0,
                 tasty=0,
                 vibe=0,
                 kind=0,
                ):
        self.id = id
        self.contents = contents
        self.point = point
        
        self.photoCnt = photoCnt
        self.likeCnt = likeCnt
        self.kakaoMapUserId = kakaoMapUserId
        
        self.username=username
        self.photoList = photoList
        #self.strengths = strengths
        self.userCommentCount = userCommentCount
        
        self.userCommentAverageScore = userCommentAverageScore
        self.date = date
        self.store_id = store_id

        # 장점
        self.effective = effective
        self.clean = clean
        self.tasty = tasty
        self.vibe = vibe
        self.kind = kind

        
    def __repr__(self):
       return f"not implemented"
