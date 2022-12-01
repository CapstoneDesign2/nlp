※cmd는 윈도우 cmd이다.

1. 도커 파일 빌드

처음에 windows docker를 설치한다.

이후에 docker 파일이 있는 directory로 이동한다. 이 프로젝트에서는 도커파일이라는 폴더다.

해당 디렉토리에서 cmd를 켜고 docker build . -t capstone 이라는 명령어를 실행한다.

의미는 docker build (현재 디렉토리) -t (빌드 파일의 이름) 이라는 의미

2. 도커 실행

도커 이미지를 생성한 이후에는 cmd로 들어가서 

docker run -it --rm  --gpus all -v C:\Users\(사용자 이름)\capstone:/workspace capstone 이 명령어를 실행한다.

의미는
          가동    내부접근      도커 종료시 삭제    gpu 전부 사용 
docker run      -it       --rm                  --gpus all -v    (현재 컴퓨터에서 연결할 디렉토리):(도커 내부에서 디렉토리) (사용할 이미지 이름)

고로 여기서 수정할 곳은 C:\Users\(사용자 이름)\capstone 이 부분을 현재 컴퓨터의 디렉토리로 변경하면 된다.

3. 도커 접속 이후

도커에 접속을 하면 리눅스와 동일하게 model 디렉토리로 이동후에 트레이닝을 시작하면 된다.

도커 내부에서 bash script.sh라는 명령어를 실행해서 학습을 실행하면 된다.



※근데 그냥 도커 설치를 못하겠다 싶으면 해당 pip 페키지를 설치하면 된다. 파이 토치는 컴퓨터의 버젼에 맞게!
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip install pytorch_lightning && \
    pip install transformers && \
    pip install sklearn && \
    pip install emoji && \
    pip install pandas && \
    pip install soynlp && \
    pip install pandas && \
    pip install numpy && \
    pip install requests && \
    pip install python-dotenv && \
    pip install pymysql && \
    pip install python-telegram-bot && \
    pip install python-dotenv && \
    pip install SQLAlchemy


