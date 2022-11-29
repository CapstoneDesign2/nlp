#!/bin/bash

: << "END"
#parser.add_argument('--model_task', type=str, default='multi_task_model', help='single_task_model 선택 혹은 multi_task_model') # 이건 필수
#parser.add_argument('--pretrained_model', type=str, default='beomi/kcbert-large', help='type of model')
#parser.add_argument('--batch_size', type=int, default=4, help='size of batch')
#parser.add_argument('--lr', type=float, default=5e-6, help='number of learning rate')
#parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
#parser.add_argument('--train_data_path', type=str, default='./data/train.tsv', help='train file path')
#parser.add_argument('--val_data_path', type=str, default='./data/valid.tsv', help='validation file path')
#parser.add_argument('--test_mode', type=bool, default=False, help='whether to turn on test')
#parser.add_argument('--optimizer', type=str, default='AdamW', help='type of optimizer')
#parser.add_argument('--lr_scheduler', type=str, default='exp', help='type of learning scheduler')
#parser.add_argument('--sensitive', type=int, default=0, help='how sensitive 0이면 sensitive 하기 1 이면 둔감')
#parser.add_argument('--test_name', type=str, default='no_name', help='실험 이름 / directory 로 사용한다')
END

#변수 이름 지정
# 이번에 조절할꺼 model_task lr pretrarined_model sensitive test_name
# 모델은 kcbert / kcelectra / kobert

# 모델 이름의 array
# 폴더 이름으로 사용할 문자열
# lr 
# sensitive
# model_task

#python3 trainning.py --model_task multi_task_model --pretrained_model beomi/KcELECTRA-base --lr 0.000005 --sensitive 1 --test_name multi_kcelec_0.000005_1

python3 message.py --command "start"

pretrained_model_list=("beomi/kcbert-large" "beomi/KcELECTRA-base")
pretrained_model_str=("kcbert-l" "kcelec")
train_path=("../data/train.tsv" "../data/enhanced_train.tsv")
#valid_path=("../data/valid.tsv" "../data/enhanced_valid.tsv")
lr_list=(0.000001)

NOW_TEST_NUMBER=1
TOTAL_TEST_NUMBER=`expr ${#pretrained_model_list[@]} \* ${#lr_list[@]} \* 2`

# electra에서는 multi-task model 작동하지 않음
# 이번에 들어가는 인수 {model_task} {lr} {pretrarined_model} {sensitive} {test_name} 총 5개

for (( j = 0 ; j < ${#pretrained_model_list[@]} ; j++ ))  ; do
    for lr in "${lr_list[@]}" ; do
        for (( i = 0 ; i < ${#train_path[@]} ; i++ ))  ; do
START=$(date +%s)

#python3 message.py \
#--command \
#"python3 trainning.py\
#--pretrained_model ${pretrained_model_list[$j]} \
#--lr ${lr} \
#--now_number $NOW_TEST_NUMBER \
#--total_number $TOTAL_TEST_NUMBER \
#"

echo "
python3 trainnig.py \
--pretrained_model ${pretrained_model_list[$j]} \
--lr ${lr} \
--train_data_path ${train_path[$i]} \
"

#python3 trainning.py \
#--pretrained_model ${pretrained_model_list[$j]} \
#--lr ${lr} \
#--train_data_path ${train_path[$i]} \
#END=$(date +%s)
#DIFF=$(( $END - $START ))

python3 message.py \
--command \
"python3 trainning.py\
--pretrained_model ${pretrained_model_list[$j]} \
--lr ${lr} \
--now_number $NOW_TEST_NUMBER \
--total_number $TOTAL_TEST_NUMBER \
--time_elapsed $DIFF
"
NOW_TEST_NUMBER=$(($NOW_TEST_NUMBER + 1))
        done
    done
done

python3 message.py --command "done!"