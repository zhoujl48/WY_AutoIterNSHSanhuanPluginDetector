#!/usr/bin/env bash
# Usage_1: bash predict_daily.sh

# 工作目录
WORK_DIR=/home/zhoujialiang/online_sanhuan

# 定义参数
ds_pred=`date -d "-1 days" +%Y%m%d`                 # 预测日期
last_friday=`date -d "friday -1 weeks" +%Y%m%d`     # 过去最近的周五
ds_start=`date -d "$last_friday -37 days" +%Y%m%d`  # 滑窗模型数据样本开始日期
ds_to_delete=`date -d "-15 days" +%Y%m%d`           # 需删除的数据日期


# 全量样本id
echo /usr/bin/python3 $WORK_DIR/get_ids.py total --ds_start $ds_pred --ds_num 1 &&
/usr/bin/python3 $WORK_DIR/get_ids.py total --ds_start $ds_pred --ds_num 1 &&

# 拉取数据序列
echo /usr/bin/python3 $WORK_DIR/dataloader.py --ds_start $ds_pred --ds_num 1 &&
/usr/bin/python3 $WORK_DIR/dataloader.py --ds_start $ds_pred --ds_num 1 &&

# 模型预测
echo /usr/bin/python3 $WORK_DIR/blstm_predict.py $ds_pred --ds_start $ds_start --ds_num 28 &&
/usr/bin/python3 $WORK_DIR/blstm_predict.py $ds_pred --ds_start $ds_start --ds_num 28 &&

# 关联画像
echo /usr/bin/python3 $WORK_DIR/huaxiang_join.py $ds_pred &&
/usr/bin/python3 $WORK_DIR/huaxiang_join.py $ds_pred &&

# 删除多余旧数据
echo /usr/bin/python $WORK_DIR/delete_data.py --date $ds_to_delete &&
/usr/bin/python $WORK_DIR/delete_data.py --date $ds_to_delete