import csv

headers = ['_id','created_at','screen_name','favorite_count','retweet_count','text','source','country_code','location','latitude','longitude']
rows = []

with open('data/percocet.csv') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        raw_text = row[5]
        index = raw_text.find('RT')
        if index < 0:
            rows.append(row)

with open('data/percocet_noRT.csv','w+') as f2:
    f2_csv = csv.writer(f2)
    f2_csv.writerow(headers)
    f2_csv.writerows(rows)    
   

