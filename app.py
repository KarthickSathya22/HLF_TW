import tabula
import dateutil
import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)

#This is a function to read the pdf bank statement of HDFC Bank:
def HDFC_PDF(file):
    """
    This function wil parse the HDFC PDF Statement.
    This function will return the average bank balance per month.
    """
    #Read PDF file:
    tables = tabula.read_pdf(file,pages='all')

    #Combining all tables:
    table = []
    for i in range(len(tables)):
        s1 = tables[i].values.tolist()
        table.extend(s1)


    #Removing unwanted columns:
    ex = []
    for i in range(len(tables)):
        if tables[i].shape[1] == 7:
            ex.extend(tables[i].values.tolist())
        elif tables[i].shape[1] == 6:
            table = tables[i].values.tolist()
            for i in table:
                i.append(i[5])
                i[5] = np.nan
                ex.append(i)

        elif tables[i].shape[1] == 8:
            table = tables[i].values.tolist()
            for i in table:
                del i[2]
                ex.append(i)

    #Creating dataframe:
    df = pd.DataFrame(ex,columns=['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.','Deposit Amt.', 'Closing Balance'])

    #Removing rows which having date is null:
    df = df[~df['Date'].isnull()]

    #Parsing Closing Price
    df["Closing Balance"] = df["Closing Balance"].astype(str)

    #Converting dataset into List:
    l1 = df.values.tolist()

    #Handiling Closing Balance column:
    final = []
    for i in l1:
        splits = (i[-1].split())
        if (len(splits)>1):
            i[-2] = splits[0]
            i[-1] = splits[1]
            final.append(i)
        else:
            final.append(i)

    #Creating dataframe:
    final = pd.DataFrame(final,columns=['Date', 'Narration', 'Chq/Ref.No', 'Value Dt', 'Withdrawal Amt','Deposit Amt', 'Closing Balance'])

    #Parsing the date fields:
    final['Date'] = final['Date'].apply(dateutil.parser.parse, dayfirst=True)
    final['Value Dt'] = final['Value Dt'].apply(dateutil.parser.parse, dayfirst=True)

    #Paring prices:
    final['Closing Balance'] = final['Closing Balance'].astype(str)
    col = ['Closing Balance']
    for i in col:
        val = []
        for j in final[i]:
            val.append(''.join(j.split(',')))
        final[i] = val

    #TypeCasting Closing Balance:
    col = ['Closing Balance']
    for i in col:
        final[i] = pd.to_numeric(final[i],errors='coerce')

    #Group by operation to close price:
    group = final.groupby(pd.Grouper(key='Date',freq='1M'))

    #Filtering close balance per month:
    balance_month = []
    for i in group:
        a = i[1]
        balance_month.append(a['Closing Balance'].iloc[-1])

    #Closing Balance Per Month:
    return np.average(balance_month)


#Loading a model:
model = pickle.load(open('model_cv_iso.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/back',methods=['POST','GET'])
def back():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    predict_request = []
    res = []
    
    status = request.form["martial_status"]
    married = {2750:"Married",2751:"Un Married"}
    predict_request.append(status)
    res.append(married.get(int(status)))
    
    dep = request.form["dependants"]
    predict_request.append(dep)
    res.append(dep)
    
    resi = request.form["residence"]
    residence = {2755:"Own",2756:"Rent"}
    predict_request.append(resi)
    res.append(residence.get(int(resi)))
    
    year = request.form["staying_year"]
    predict_request.append(year)
    res.append(year)
    
    #Uploading file:
    file = request.files['file']
    filename = file.filename
    extn = filename.split('.')[-1]   
    destination = file
  
    #Checking for extension of file: 
    if (extn.casefold() == 'pdf'):
        #Returned a result from a function calling:
        clobal =  HDFC_PDF(destination)
    
    if (extn.casefold() == 'xls'):
        #Loading dataset:
        df = pd.read_excel(destination)
        
        #Fetching transactions only:
        row_no = 0
        for i in df.iloc[:,0]:
            if i == 'Date':
                df = df.iloc[row_no:]
                break
            row_no = row_no+1
        
        #Set a features name:
        df.columns = ['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.','Deposit Amt.', 'Closing Balance']
        
        #Reset the Index:
        df.reset_index(drop=True, inplace=True)
        
        #Dropping first two records:
        df.drop([0,1],axis=0,inplace=True)
        
        #Reset the Index:
        df.reset_index(drop=True, inplace=True)
        
        row_no = 0
        for i in df['Date']:
            if len(str(i)) != 8:
                df = df.iloc[:row_no]
                break
            row_no = row_no + 1
            
        # Parsing date:
        df['Date'] = df['Date'].apply(dateutil.parser.parse, dayfirst=True)
        table = df
        
        #Group by operation to find opening and close price:
        group = table.groupby(pd.Grouper(key='Date',freq='1M'))
        
        #Filtering open and close balance per month:
        balance_month = []
        for i in group:
            a = i[1]
            balance_month.append(a['Closing Balance'].iloc[-1])
        
        clobal = (np.average(balance_month))
   
    predict_request.append("{:.2f}".format(clobal))
    res.append("{:.2f}".format(clobal))

    asset = request.form["assetvalue"]
    predict_request.append(asset)
    res.append(asset)
    
    cat = request.form["productcat"]
    prod_cat = {1784:"LOAN AGAINST PROPERTY",
            926:"CAR",
            912:"MULTI UTILITY VEHICLE",
            945:"VIKRAM",
            1402:"TRACTOR",
            1373:"USED VEHICLES",
            1672:"TIPPER",
            1664:"FARM EQUIPMENT",
            1541:"TWO WHEELER",
            634:"INTERMEDIATE COMMERCIAL VEHICLE",
            527:"HEAVY COMMERCIAL VEHICLE",
            528:"CONSTRUCTION EQUIPMENTS",
            529:"THREE WHEELERS",
            530:"LIGHT COMMERCIAL VEHICLES",
            531:"SMALL COMMERCIAL VEHICLE",
            738:"MEDIUM COMMERCIAL VEHICLE",
            783:"BUSES"}
    predict_request.append(cat)
    res.append(prod_cat.get(int(cat)))
    
    brand = request.form["brand"]
    brand_type = {1:"Others",
                  1360:"HONDA",
                  1542:"HERO", 
                  1544:"HMSI",
                  1547:"YAMAHA",
                  1546:"SUZUKI",
                  1647:"TVS",
                  1549:"ROYAL ENFIELD"
                  }
    predict_request.append(brand)
    res.append(brand_type.get(int(brand)))
    
    indus = request.form["industrytype"]
    ind_cat = {1782:"SALARIED",1783:"SELF EMPLOYEED",603:"AGRICULTURE",
     604:"PASSENGER TRANSPORTATION",605:"CONSTRUCTION",875:"INFRASTRUCTURE",
     876:"CEMENT",877:"OIL AND GAS",878:"GOVERNMENT CONTRACT",879:"OTHERS",658:"MINE"}
    predict_request.append(indus)
    res.append(ind_cat.get(int(indus)))
    
    tenure = request.form["tenure"]
    predict_request.append(tenure)
    res.append(tenure)
    
    instal = request.form["instalcount"]
    predict_request.append(instal)
    res.append(instal)
    
    chasasset = request.form["chasasset"]
    predict_request.append(chasasset)
    res.append(chasasset)
    
    chasinitial = request.form["chasinitial"]
    predict_request.append(chasinitial)
    res.append(chasinitial)
    
    chasfin = int(chasasset) - int(chasinitial)
    predict_request.append(chasfin)
    res.append(chasfin)
    
    fininter = request.form["finaninterest"]
    predict_request.append(fininter)
    res.append(fininter)
    
    interestamount = (int(chasfin)*(int(tenure)/12)*(float(fininter)))/100
    emi = (int(chasfin)+int(interestamount))/int(tenure)
    predict_request.append(int(emi))
    res.append(int(emi))
    
    inflow = request.form["totinflow"]
    predict_request.append(inflow)
    res.append(inflow)
    
    score = request.form["score"]
    predict_request.append(score)
    res.append(score)
    
    cibil = request.form["another_score"]
    predict_request.append(cibil)
    res.append(cibil)
    
    age = request.form["age"]
    predict_request.append(age)
    res.append(age)
    
    loan = (int(chasfin)*100/int(chasasset)) 
    if (loan<85):
        loan_to_value = 120
    elif ((loan>=85) and (loan <=90)):
        loan_to_value = 100
    elif (loan>90):
        loan_to_value = 50
    predict_request.append(loan_to_value)
    res.append(loan_to_value)
    
    brand = int(brand)
    # Approved Models (Hero /Honda/Suzuki  - All Products)
    if ((brand == 1360) | (brand == 1542) | (brand == 1544) | (brand == 1546)):
        asset_finance = 100
    
    # Unapproved Models (Only Yahama, TVS and Royal Enfield Bike Variants) with PM Approval:
    elif ((brand == 1547) | (brand == 1647) | (brand == 935) | (brand == 1549)):  
        asset_finance = 50
    
    else:
        asset_finance = -100
    predict_request.append(asset_finance)
    res.append(asset_finance)
    
    gi = int(inflow)*12
    if gi > 12000:
        grossincome = 100
    elif gi > 8000 and gi<=12000:
        grossincome = 70
    elif gi >= 5000 and gi<=8000:
        grossincome = 50
    elif gi < 5000:
        grossincome = -50    
    predict_request.append(int(grossincome))
    res.append(grossincome)
     
    bank = request.form["bank_detail"]
    old = {0:"New Account",2:"<3 Month Old",4:">3 Month Old",7:">6 Month Old"}
    res.append(old.get(int(bank)))
    
    flag = False
    #>1 time ABB, >6 months old bank account and >3 yrs in Rented House:
    if ((emi > int(clobal)) & (int(bank) > 6) & ((int(resi) == 2756) & (int(year) > 3))):
        banking = 100
        flag = True
    
    #>1 time ABB & > 6 months old bank account and >2 yrs in Rented House
    if not flag:
        if ((emi > int(clobal)) & (int(bank) > 6) & ((int(resi) == 2756) & (int(year) > 2))):
            banking = 80
            flag = True
    
    # >1 time ABB & >3 months old bank account and >2 yrs in Rented House
    if not flag:
        if ((emi > int(clobal)) & (int(bank) > 3) & ((int(resi) == 2756) & (int(year) > 2))):
            banking = 60
            flag = True
    
    #ABB is less than 1 time or new bank acccount but Borrower has own house:
    if not flag:
        if ((emi > int(clobal)) | ((int(bank) == 0) & (int(resi) == 2755))):
            banking = 100
            flag = True
    
    #<3 months old bank account or <1 time ABB or <2 yrs in Rented House
    if not flag:
        if ((emi < int(clobal)) | (int(bank) < 3) | ((int(resi) == 2756) & (int(year) < 2))):
            banking = -100
            flag = True         
    predict_request.append(banking)
    res.append(banking)
    
    stability = request.form["stability"]
    stability_type = {1:"Salaried with over 1 year",
                      2:"Salaried with over 6 Months",
                      3:"Salaried less than 6 Months"}
    res.append(stability_type.get(int(stability)))
    if (int(stability) == 1):
        stab = 100
    if (int(stability) == 2):
        stab = 80
    if (int(stability) == 3):
        stab = -80    
    predict_request.append(stab)
    res.append(stab)
    
    #Age between 21 and 50 years, Married with No Dependents:
    if (((int(age) >= 21) & (int(age) <= 50)) & ((int(status) == 2750) & (int(dep) == 0))):
        age_martial = 120
    
    #Age between 21 and 50 years, Married with Dependents:
    elif (((int(age) >= 21) & (int(age) <= 50)) & ((int(status) == 2750) & (int(dep) != 0))):
        age_martial = 100
    
    #Age between 51 and 62 years, Married with No Dependents:
    elif (((int(age) >= 51) & (int(age) <= 62)) & ((int(status) == 2750) & (int(dep) == 0))):
        age_martial = 80
    
    #Age between 51 and 62 years, Married with Dependents:
    elif (((int(age) >= 51) & (int(age) <= 62)) & ((int(status) == 2750) & (int(dep) != 0))):
        age_martial = 60
    
    #Age between 21 and 40 years, Unmarried:
    elif (((int(age) >= 21) & (int(age) <= 40)) & (int(status) == 2751)):
        age_martial = 100
    
    # Age between 18 and 21, Above 62 and Age between 41 and 60 Unmarried
    elif (((int(age) >= 18) & (int(age) <= 21)) | ((int(age) >= 62)) | (((int(age) >= 41) & (int(age) <= 60)) & (int(status) == 2751))):
        age_martial = 0
    predict_request.append(age_martial)
    res.append(age_martial)
    
    geo = request.form["geo"]
    geo_type = {1:"Less than 15 Km",
                2:"More than 15 Km"}
    res.append(geo_type.get(int(geo)))
    if (int(geo) == 1):
        geo_lim = 1
    else: 
        geo_lim = 0 
    predict_request.append(geo_lim)
    res.append(geo_lim)
    
    gender_dict = {'M':[0,1],'F':[1,0]}
    cate = request.form["gender"]
    if cate == 'M':
        res.append('Male')
    else:
        res.append('Female')
    predict_request.extend(gender_dict.get(cate))
    predict_request = list(map(float,predict_request))
    predict_request = np.array(predict_request)
    prediction = model.predict_proba([predict_request])[0][-1]
    output = int((1 - prediction)*100)
    if output < 50:
        condition = 'Risky'
    if output >= 50 and output <= 69:
        condition = 'Barely Acceptable'
    if output >= 70 and output <=89:
        condition = 'Medium'
    if output >= 90 and output <= 99:
        condition = 'Good'
    if output == 100:
        condition = 'Superior'
    return render_template('resultpage.html', prediction_text=output,data=res,status=condition)

if __name__ == "__main__":
    app.run(debug=True)
