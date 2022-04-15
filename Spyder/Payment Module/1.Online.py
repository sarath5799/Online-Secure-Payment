import pandas as pd
from datetime import date, datetime

today=date.today()
d1 = today.strftime("%d-%m-%Y")

now=datetime.now()
dt_string = now.strftime("%H:%M:%S")

customer=pd.read_csv('data/Customer_bank.csv')

Retailer=pd.read_csv('data/Retailer_bank.csv')

shopnumber=input("Enter Shop Number:")

yournumber=input("Enter yournumber Number:")

EnterAmount=input("Enter amount to be paid:")

detectedname=input("Enter your name:")


ind=customer[customer['Mobile_num']==int(yournumber)].index[0]
namemobcheck=customer[customer['Mobile_num']==int(yournumber)]['Name'][ind]

if(detectedname==namemobcheck):
    print("Same name")
    cusbalance=customer[customer['Mobile_num']==int(yournumber)]['Balance'][ind]
    if(int(cusbalance)>int(EnterAmount)):
        idx_R=Retailer[Retailer['Mobile_num']==int(shopnumber)].index[0]
        idx_RR=Retailer[Retailer['Mobile_num']==int(shopnumber)].index
        retbalance=Retailer[Retailer['Mobile_num']==int(shopnumber)]['Balance'][idx_R]
        retbalance=retbalance+int(EnterAmount)
        Retailer.loc[idx_RR,'Balance']= retbalance
        Retailer.loc[idx_RR,'UpdatedDate']=d1 + dt_string
        print("Transaction Successfull")
        cusbalance=cusbalance-int(EnterAmount)
        idx_C=customer[customer['Mobile_num']==int(yournumber)].index
        customer.loc[idx_C,'Balance']= cusbalance
        customer.loc[idx_C,'updatedDate']= d1 + dt_string
        print("amount detected from customer")
        customer.to_csv("data/Customer_bank.csv", index=False)
        Retailer.to_csv("data/Retailer_bank.csv",index=False)
        
    else:
        print("Low Balance")
        
   
else:
    print("Not same name")
    

