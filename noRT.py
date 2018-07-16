import csv

headers = ['_id','created_at','screen_name','favorite_count','retweet_count','text,source','country_code','location','latitude','longitude']
rows = []

with open('codeine.csv') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        raw_text = row[5]
        index = max(raw_text.find('RT'),raw_text.find('codeine crazy'),raw_text.find('Codeine Crazy'),raw_text.find('Codeine crazy'))
        print(index)
        if index < 0:
            rows.append(row)

with open('codeine_noRT.csv','w+') as f2:
    f2_csv = csv.writer(f2)
    f2_csv.writerow(headers)
    f2_csv.writerows(rows)    
   

