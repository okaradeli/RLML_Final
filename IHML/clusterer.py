import pandas as pd

CLUSTER_COLUMNS = ["loan_amnt","installment","annual_inc","open_acc","pub_rec","total_acc","acc_now_delinq","tot_coll_amt","acc_open_past_24mths","avg_cur_bal","mort_acc","pub_rec_bankruptcies","tot_hi_cred_lim","total_bal_ex_mort","total_bc_limit","total_il_high_credit_limit","term_num","isPresident","isDirector","annual_inc_both"]
K_CLUSTER=4
FILE="excel_experiments/all_model_out_v11_knn_CSV.csv"
FILE_OUT=FILE+"_OUT.xlsx"
df = pd.read_csv(FILE,delimiter=";")
df1=df[CLUSTER_COLUMNS]

##find the significant clusters
print("started JOB")
from sklearn.cluster import KMeans
km = KMeans(n_clusters=K_CLUSTER, init='k-means++', n_init=10)
km.fit(df1)
pred_clusters = km.predict(df1)
print("persisting to excel")
##persist to Excel
df["CLUSTER"]=pred_clusters
df.to_excel(FILE_OUT)

print("JOB COMPLETE")


