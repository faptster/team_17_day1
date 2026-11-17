# TGR2026 Edge AI Track 2 · Day 1

ระบบจำแนกภาพเมล็ดกาแฟหนึ่งคลาสต่อภาพด้วย YOLO26m-cls บน Raspberry Pi 4 ด้วย CPU

## ติดตั้งและทำนาย

แนะนำ Raspberry Pi OS 64 บิต เปิด Terminal ที่โฟลเดอร์ซึ่งมี `infer.py` แล้วติดตั้งด้วย Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir --timeout 120 --resume-retries 10 -r requirements.txt
```

เมื่อวางภาพทดสอบใน `test/` ให้รันคำสั่งมาตรฐาน

```bash
python infer.py --input test/ --weights weights/ --out outputs/
```

- `--input` รับไฟล์หรือโฟลเดอร์ รองรับ `.jpg`, `.jpeg` และ `.png` โดยไม่สนตัวพิมพ์เล็กหรือใหญ่
- `--weights` รับโฟลเดอร์ สคริปต์โหลด `best.pt` จากโฟลเดอร์นี้ หากใช้ชื่ออื่น ให้มีไฟล์ `.pt` เพียงไฟล์เดียว
- `--out` คือโฟลเดอร์ผลลัพธ์ ให้ใช้แยกจากภาพทดสอบและโมเดล สคริปต์เขียน CSV แบบ UTF-8 ที่ `outputs/predictions.csv` หัวตาราง `filename,class` และคัดลอกภาพต้นฉบับไปยังโฟลเดอร์คลาสที่ทำนาย
- ZIP มีโครง `outputs/` สำหรับรอผลจริง เมื่อรันคำสั่งแล้วระบบจะสร้างหนึ่งแถวต่อภาพ

## โมเดลและการเตรียมภาพ

`weights/best.pt` คือ checkpoint YOLO26m-cls แบบ 7 คลาสจากขั้น fine-tune สุดท้าย ใช้ไฟล์นี้เพียงไฟล์เดียวในการทำนาย

ระบบปรับทิศตาม EXIF แปลงภาพเป็น RGB ย่อภาพทั้งเฟรมแบบรักษาสัดส่วนด้วย bilinear แล้วเติมขอบสีเทา `(220, 220, 220)` ให้ครบ 320 × 320 พิกเซล จากนั้นใช้การแปลงข้อมูลมาตรฐานของ classifier คะแนนอันดับหนึ่งเป็นคลาสของภาพ โดยปิดการทำ test-time augmentation

## เทรนซ้ำ

ใช้ `train.py` กับโฟลเดอร์ข้อมูล `day_1` ที่แตกจาก ZIP ของผู้จัดแล้ว สคริปต์ใช้ภาพ `grid` และ `circle` สร้างข้อมูลฝึก 154 ภาพ และกัน `square` แยกจากข้อมูลฝึก สคริปต์รองรับโฟลเดอร์ข้อมูลที่เตรียมไว้แล้วและมี `train/<class>/` ด้วย

```bash
python -m pip install scipy
python train.py --data /path/to/day_1 --device 0 --out weights_retrained/
```

`--device 0` ใช้ GPU สำหรับฝึก ส่วน `--device cpu` ใช้ CPU การฝึกเริ่มจาก `yolo26m-cls.pt` ของ Ultralytics ตรึง backbone 5 epochs ที่ learning rate 0.0005 จากนั้นปลดทุกชั้น 25 epochs ที่ learning rate 0.0001 แล้วบันทึก checkpoint สุดท้ายเป็น `best.pt` ในโฟลเดอร์ที่กำหนด
