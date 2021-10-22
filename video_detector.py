import cv2, time, pandas
from datetime import datetime

first_frame = None
status_list = [None,None]
collect_time = []
df = pandas.DataFrame(columns=['Start','End'])

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_frame = cv2.threshold(delta_frame, 75, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame,None,iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1
        (x, y, w, h)= cv2.boundingRect(contour)
        reci = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    status_list.append(status)

    status_list=status_list[-2:]


    if status_list[-1] == 1 and status_list[-2] == 0:
        collect_time.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        collect_time.append(datetime.now())

    cv2.imshow('Gray Frame', gray)
    cv2.imshow('Delta Frame', delta_frame)
    cv2.imshow('Theshold Frame', thresh_frame)
    cv2.imshow('Color Frame', frame)

    key = cv2.waitKey(1)
    
    if key == ord('q'):
        if status == 1:
            collect_time.append(datetime.now())
        break
#print(status_list)
for x in range(len(collect_time)):
    print(x,'/',len(collect_time),')',collect_time[x])

for x in range(0,len(collect_time),2):
    df = df.append({"Start":collect_time[x],"End":collect_time[x+1]},ignore_index=True)

df.to_csv("object_detected_collect_time.csv")

video.release()
cv2.destroyAllWindows


