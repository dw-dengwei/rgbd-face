time_stamp=`date +'%Y-%m-%d-%T'`
cat config/config.py > log/${time_stamp}.config
python main.py 1>log/${time_stamp}.stdout 2>log/${time_stamp}.stderr
