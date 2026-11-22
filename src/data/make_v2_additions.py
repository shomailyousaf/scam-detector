
import random
from pathlib import Path

import pandas as pd

OUT = Path("data/raw/scam_messages_v2_additions.csv")
SOURCE = "synthetic_v2_generic"
SEED = 42
VARIANTS = 3
FIRST_ID = 100001

SLOTS = {
    "name": ["Ali", "Ahmed", "Sana", "Ayesha", "Bilal", "Usman", "Hina", "Fatima", "Zain", "Maryam"],
    "rel": ["Ammi", "Abbu", "Bhai", "Baji", "Khala", "Chachu", "Mamu", "Phuppo"],
    "amt": ["2,000", "5,000", "7,500", "10,000", "15,000", "20,000", "25,000", "50,000"],
    "bank": ["HBL", "Meezan Bank", "UBL", "MCB", "Allied Bank", "Bank Alfalah"],
    "wallet": ["JazzCash", "Easypaisa"],
    "courier": ["TCS", "Leopards", "M&P", "Trax", "PostEx"],
    "code": ["482-913", "305-771", "619-204", "774-038", "521-660"],
    "last4": ["4417", "9032", "1186", "7745", "3309"],
    "time": ["5 baje", "shaam 7 baje", "kal subah", "Jummah ke baad", "raat 9 baje"],
    "city": ["Lahore", "Karachi", "Islamabad", "Multan", "Faisalabad", "Peshawar"],
    "badlink": ["bit.ly/hbl-verify", "tinyurl.com/wallet-bonus", "secure-pk-verify.xyz/login",
                "pak-parcel-track.top/pay", "claim-reward-now.click/pk"],
    "order": ["4471", "8820", "1934", "5566", "7012"],
}


T = [
    
    ("whatsapp_takeover", "SCAM", "scam", "Roman-Urdu", "{name} galti se aapke number par mera WhatsApp code chala gaya hai, please woh 6 digit code wapas bhej do"),
    ("whatsapp_takeover", "SCAM", "scam", "Roman-Urdu", "Assalam o alaikum, main {name} hun, naya phone liya hai. Aapko jo code aya hai woh jaldi forward kar dein"),
    ("whatsapp_takeover", "SCAM", "scam", "Roman-Urdu", "WhatsApp support: aapka account 24 ghante mein band ho jayega, verification ke liye SMS wala code reply karein"),
    ("whatsapp_takeover", "SCAM", "scam", "Code-Mixed", "Hi {name}, urgent hai, mera account logout ho gaya. Ek verification code tumhare number pe aya hoga, please send kar do"),
    ("whatsapp_takeover", "SCAM", "scam", "English", "WhatsApp Security: unusual login detected. Reply with the 6-digit code you just received to keep your account active"),
    ("whatsapp_takeover", "SCAM", "scam", "English", "Hey it's {name}, I sent you a code by mistake. Can you send it back to me quickly? It's urgent"),
    ("whatsapp_takeover", "SCAM", "scam", "Roman-Urdu", "Family group mein add karne ke liye code chahiye, jo abhi SMS aya hai woh mujhe bhej do {rel}"),
    ("whatsapp_takeover", "SCAM", "scam", "Urdu", "آپ کے نمبر پر غلطی سے میرا واٹس ایپ کوڈ چلا گیا ہے، براہ کرم وہ کوڈ واپس بھیج دیں"),
    ("whatsapp_takeover", "SCAM", "scam", "Roman-Urdu", "Aapka WhatsApp verify nahi hua, is link par ja kar code daalein warna number block: {badlink}"),
    ("whatsapp_takeover", "SCAM", "scam", "Code-Mixed", "Bro mera WhatsApp hack ho gaya tha, recovery ke liye tumhara code chahiye, sirf ek minute ka kaam hai"),
    ("whatsapp_takeover", "NORMAL", "legit", "Roman-Urdu", "{name} maine naya number le liya hai, purana band kar diya hai. Ye wala save kar lo"),
    ("whatsapp_takeover", "NORMAL", "legit", "English", "I switched to a new phone, my WhatsApp chats are restored now. Talk later"),
    ("whatsapp_takeover", "NORMAL", "legit", "Roman-Urdu", "Group mein add kar diya hai {rel}, ab sab ki photos wahan share karna"),
    ("whatsapp_takeover", "NORMAL", "legit", "Code-Mixed", "WhatsApp update kar lo, naya version mein voice note fast ho gaya hai"),
    ("whatsapp_takeover", "NORMAL", "hardneg", "English", "Your WhatsApp code: {code}. Don't share this code with anyone"),
    ("whatsapp_takeover", "NORMAL", "hardneg", "Roman-Urdu", "Koi bhi aapse WhatsApp ka code maange to mat dena, chahe woh dost hi kyun na lage"),
    ("whatsapp_takeover", "NORMAL", "hardneg", "Roman-Urdu", "{rel} kisi ko bhi SMS wala code mat bhejna, aaj kal is tarah account hack ho rahe hain"),
    ("whatsapp_takeover", "NORMAL", "hardneg", "Code-Mixed", "Two-step verification on kar lo settings mein, phir koi tumhara WhatsApp hack nahi kar sakta"),
    ("whatsapp_takeover", "NORMAL", "hardneg", "Urdu", "اپنا واٹس ایپ کوڈ کسی کو نہ بتائیں، کمپنی کبھی کوڈ نہیں مانگتی"),
    ("whatsapp_takeover", "NORMAL", "hardneg", "English", "Reminder: our office will never ask for your WhatsApp or SMS verification code"),


    ("emergency_family_scam", "SCAM", "scam", "Roman-Urdu", "{rel} ye mera naya number hai, purana kharab ho gaya. Abhi {amt} rupay {wallet} kar do, bohat urgent hai"),
    ("emergency_family_scam", "SCAM", "scam", "Roman-Urdu", "Beta main hospital mein hun, phone kisi aur ka hai. {amt} isi number par bhej do, baad mein baat karta hun"),
    ("emergency_family_scam", "SCAM", "scam", "Roman-Urdu", "Main {name} bol raha hun, police ne bike pakad li hai, {amt} abhi bhejo warna challan double ho jayega"),
    ("emergency_family_scam", "SCAM", "scam", "Code-Mixed", "Hi {rel}, new number hai mera. Kisi ko mat batana, please {amt} transfer kar do, emergency hai"),
    ("emergency_family_scam", "SCAM", "scam", "English", "Hi Mum, this is my new number. My phone broke. Can you send {amt} urgently? I'll explain later"),
    ("emergency_family_scam", "SCAM", "scam", "Roman-Urdu", "Aapka beta accident mein zakhmi hai, operation ke liye {amt} fauran is account mein jama karwain"),
    ("emergency_family_scam", "SCAM", "scam", "Urdu", "امی یہ میرا نیا نمبر ہے، فوراً {amt} روپے بھیج دیں، بہت ضروری ہے"),
    ("emergency_family_scam", "SCAM", "scam", "Roman-Urdu", "{name} dubai se bol raha hun, yahan phas gaya hun, ticket ke liye {amt} chahiye, kisi ko batana mat"),
    ("emergency_family_scam", "SCAM", "scam", "Code-Mixed", "Urgent: mera wallet chori ho gaya, {wallet} pe {amt} bhej do, kal return kar dunga promise"),
    ("emergency_family_scam", "SCAM", "scam", "Roman-Urdu", "Uncle main {name} ka dost hun, woh hospital mein hai, abhi {amt} bhej dein, woh baat nahi kar sakta"),
    ("emergency_family_scam", "NORMAL", "legit", "Roman-Urdu", "{rel} ne kaha hai {time} sab ghar aa jana, khana saath khayenge"),
    ("emergency_family_scam", "NORMAL", "legit", "Roman-Urdu", "Beta pohanch gaye ho? {city} ka mausam kaisa hai wahan"),
    ("emergency_family_scam", "NORMAL", "legit", "Code-Mixed", "{name} ki birthday hai kal, cake order kar diya hai, tum {time} aa jana"),
    ("emergency_family_scam", "NORMAL", "legit", "English", "Reached {city} safely, will call you after dinner"),
    ("emergency_family_scam", "NORMAL", "legit", "Roman-Urdu", "{rel} dawai le li? Doctor ne kaha tha subah shaam leni hai"),
    ("emergency_family_scam", "NORMAL", "legit", "Urdu", "{time} سب گھر آ جانا، امی نے بریانی بنائی ہے"),
    ("emergency_family_scam", "NORMAL", "legit", "Roman-Urdu", "Maine {amt} tumhare account mein bhej diye hain fees ke liye, check kar lena"),
    ("emergency_family_scam", "NORMAL", "legit", "Code-Mixed", "Bhai mera phone kharab hai, ye {name} ka number hai, ghar aa ke baat karta hun"),
    ("emergency_family_scam", "NORMAL", "hardneg", "Roman-Urdu", "Agar koi naye number se paise maange to pehle purane number par call karke confirm karna"),
    ("emergency_family_scam", "NORMAL", "hardneg", "English", "Scam alert: fraudsters pretend to be relatives with a new number and ask for urgent money. Always call back first"),
    ("emergency_family_scam", "NORMAL", "hardneg", "Roman-Urdu", "{rel} koi bhi 'beta hospital mein hai paise bhejo' wala message aye to ghabrana nahi, pehle mujhe call karna"),
    ("emergency_family_scam", "NORMAL", "hardneg", "Urdu", "اگر کوئی نئے نمبر سے پیسے مانگے تو پہلے پرانے نمبر پر کال کر کے تصدیق کریں"),

    # ---------------- Bank phishing vs real bank alerts ----------------
    ("bank_phishing", "SCAM", "scam", "Roman-Urdu", "{bank}: aapka account aaj block ho jayega. Abhi is link par CNIC aur card details verify karein {badlink}"),
    ("bank_phishing", "SCAM", "scam", "English", "Dear customer, your {bank} card ending {last4} is suspended. Verify your PIN at {badlink} within 2 hours"),
    ("bank_phishing", "SCAM", "scam", "Roman-Urdu", "Main {bank} head office se bol raha hun, aapke account ki security update ke liye OTP bata dein"),
    ("bank_phishing", "SCAM", "scam", "Code-Mixed", "{bank} alert: suspicious transaction of Rs {amt}. Cancel karne ke liye apna card number aur CVV reply karein"),
    ("bank_phishing", "SCAM", "scam", "Roman-Urdu", "Mubarak ho! {bank} ki taraf se aapko {amt} cashback mila hai, claim ke liye ATM PIN bhejein"),
    ("bank_phishing", "SCAM", "scam", "Urdu", "آپ کا {bank} اکاؤنٹ بند ہونے والا ہے، فوراً اس لنک پر تصدیق کریں {badlink}"),
    ("bank_phishing", "SCAM", "scam", "English", "Your account will be closed today due to incomplete biometric. Update now: {badlink}"),
    ("bank_phishing", "SCAM", "scam", "Roman-Urdu", "SBP notice: aapke account mein {amt} pending hain, release ke liye {amt} fee is number par bhejein"),
    ("bank_phishing", "SCAM", "scam", "Code-Mixed", "Card renewal ke liye apna expiry date aur 3 digit code is number pe SMS karein, warna card block"),
    ("bank_phishing", "SCAM", "scam", "Roman-Urdu", "Aapka {bank} debit card hack ho gaya hai, bachane ke liye jo code abhi aya hai woh hamein batayein"),
    ("bank_phishing", "NORMAL", "legit", "English", "{bank}: Rs {amt} debited from your account ending {last4} on POS. Not you? Call the number on the back of your card"),
    ("bank_phishing", "NORMAL", "legit", "English", "{bank}: Rs {amt} credited to your account ending {last4}. Available balance updated"),
    ("bank_phishing", "NORMAL", "legit", "Roman-Urdu", "Salary aa gayi hai {bank} mein, kal bills pay kar dena"),
    ("bank_phishing", "NORMAL", "legit", "English", "Your {bank} e-statement for this month is now available in the mobile app"),
    ("bank_phishing", "NORMAL", "legit", "Code-Mixed", "{bank} branch {time} band ho jati hai, cheque jama karwana hai to jaldi chale jana"),
    ("bank_phishing", "NORMAL", "legit", "Roman-Urdu", "Maine {bank} app se {amt} transfer kar diye hain, screenshot bhej raha hun"),
    ("bank_phishing", "NORMAL", "legit", "Urdu", "آپ کے اکاؤنٹ میں {amt} روپے جمع ہو گئے ہیں۔ شکریہ {bank}"),
    ("bank_phishing", "NORMAL", "legit", "English", "Your {bank} card ending {last4} will expire next month. Collect the new card from your branch"),
    ("bank_phishing", "NORMAL", "hardneg", "English", "{bank} will never ask for your PIN, CVV or OTP by call or SMS. Report suspicious messages"),
    ("bank_phishing", "NORMAL", "hardneg", "Roman-Urdu", "Bank kabhi bhi phone par PIN ya OTP nahi poochta, koi pooche to foran call kaat dein"),
    ("bank_phishing", "NORMAL", "hardneg", "Roman-Urdu", "{rel} agar {bank} ke naam se link aye to us par click mat karna, branch jakar pooch lena"),
    ("bank_phishing", "NORMAL", "hardneg", "Code-Mixed", "Your OTP for {bank} login is {code}. Ye code kisi ke sath share na karein"),
    ("bank_phishing", "NORMAL", "hardneg", "Urdu", "بینک کبھی بھی فون پر پن یا او ٹی پی نہیں پوچھتا، ہوشیار رہیں"),
    ("bank_phishing", "NORMAL", "hardneg", "English", "Fraud alert from {bank}: do not click links asking you to verify your account. Use only the official app"),

    # ---------------- Parcel / delivery ----------------
    ("parcel_delivery_scam", "SCAM", "scam", "Roman-Urdu", "{courier}: aapka parcel customs mein ruka hai, {amt} fee is link par jama karwain warna return {badlink}"),
    ("parcel_delivery_scam", "SCAM", "scam", "English", "Your {courier} package could not be delivered. Pay Rs 150 redelivery fee here: {badlink}"),
    ("parcel_delivery_scam", "SCAM", "scam", "Roman-Urdu", "Aapka international parcel aya hai jis mein gift hai, release ke liye {amt} {wallet} karein"),
    ("parcel_delivery_scam", "SCAM", "scam", "Code-Mixed", "Rider bol raha hun {courier} se, parcel deliver karne ke liye jo OTP aya hai woh bata dein"),
    ("parcel_delivery_scam", "SCAM", "scam", "Urdu", "آپ کا پارسل کسٹمز میں رکا ہوا ہے، {amt} روپے فیس فوراً جمع کروائیں"),
    ("parcel_delivery_scam", "SCAM", "scam", "Roman-Urdu", "Address incomplete hai, parcel wapas ja raha hai. Is link par address aur card details update karein {badlink}"),
    ("parcel_delivery_scam", "SCAM", "scam", "English", "Final notice: your parcel will be destroyed in 24 hours. Confirm payment details now {badlink}"),
    ("parcel_delivery_scam", "SCAM", "scam", "Code-Mixed", "Congratulations, aapka lucky draw parcel {courier} pe hai, delivery charges {amt} advance bhejein"),
    ("parcel_delivery_scam", "NORMAL", "legit", "English", "{courier}: your shipment #{order} is out for delivery today. Please keep cash ready for COD"),
    ("parcel_delivery_scam", "NORMAL", "legit", "Roman-Urdu", "Aapka order #{order} dispatch ho gaya hai, {city} mein 2 se 3 din mein deliver hoga"),
    ("parcel_delivery_scam", "NORMAL", "legit", "Roman-Urdu", "Rider gate par hai, parcel le lein, COD Rs {amt} hai"),
    ("parcel_delivery_scam", "NORMAL", "legit", "Code-Mixed", "Parcel mil gaya hai, packing bohat achi thi, thank you"),
    ("parcel_delivery_scam", "NORMAL", "legit", "English", "Your order #{order} has been delivered. Rate your experience in the app"),
    ("parcel_delivery_scam", "NORMAL", "legit", "Urdu", "آپ کا آرڈر {order} کل تک پہنچ جائے گا"),
    ("parcel_delivery_scam", "NORMAL", "legit", "Roman-Urdu", "{name} tumhara parcel mere ghar aa gaya hai, {time} le jana"),
    ("parcel_delivery_scam", "NORMAL", "legit", "English", "{courier}: delivery rescheduled to tomorrow as requested"),
    ("parcel_delivery_scam", "NORMAL", "hardneg", "Roman-Urdu", "{courier} kabhi bhi link ke zariye customs fee nahi maangta, aise messages ignore karein"),
    ("parcel_delivery_scam", "NORMAL", "hardneg", "English", "Delivery OTP {code}: share it only with the rider at your door after you receive the parcel"),
    ("parcel_delivery_scam", "NORMAL", "hardneg", "Roman-Urdu", "Parcel ke naam par paise maangne wale fake messages aa rahe hain, sirf official app check karein"),
    ("parcel_delivery_scam", "NORMAL", "hardneg", "Code-Mixed", "Agar koi bole parcel customs mein hai, fee bhejo, to samajh jao scam hai"),

    # ---------------- Mobile wallet ----------------
    ("mobile_wallet_phishing", "SCAM", "scam", "Roman-Urdu", "{wallet}: aap ka account band hone wala hai, MPIN aur CNIC is number par bhejein"),
    ("mobile_wallet_phishing", "SCAM", "scam", "Roman-Urdu", "Mubarak! {wallet} lucky draw mein aapka {amt} ka inaam nikla hai, claim ke liye {badlink}"),
    ("mobile_wallet_phishing", "SCAM", "scam", "Code-Mixed", "Galti se aapke {wallet} mein {amt} chale gaye hain, please wapas bhej dein, main gareeb aadmi hun"),
    ("mobile_wallet_phishing", "SCAM", "scam", "English", "{wallet} reward: you have won Rs {amt}. Share the code sent to you to receive it"),
    ("mobile_wallet_phishing", "SCAM", "scam", "Urdu", "آپ کے {wallet} اکاؤنٹ میں انعام آیا ہے، وصول کرنے کے لیے پن کوڈ بتائیں"),
    ("mobile_wallet_phishing", "SCAM", "scam", "Roman-Urdu", "BISP ki qist {amt} aapke {wallet} mein aani hai, pehle 500 rupay registration fee bhejein"),
    ("mobile_wallet_phishing", "SCAM", "scam", "Roman-Urdu", "{wallet} customer care: account verification ke liye jo 4 digit code aya hai woh bata dein"),
    ("mobile_wallet_phishing", "SCAM", "scam", "Code-Mixed", "Double money offer: {amt} bhejo {wallet} pe, 1 ghante mein double wapas milega guaranteed"),
    ("mobile_wallet_phishing", "NORMAL", "legit", "Roman-Urdu", "Maine {wallet} par {amt} bhej diye hain, mil jayein to bata dena"),
    ("mobile_wallet_phishing", "NORMAL", "legit", "English", "{wallet}: You have received Rs {amt} from {name}. Thank you for using our service"),
    ("mobile_wallet_phishing", "NORMAL", "legit", "Code-Mixed", "Bijli ka bill {wallet} se pay kar diya, receipt group mein daal di hai"),
    ("mobile_wallet_phishing", "NORMAL", "legit", "Roman-Urdu", "{name} apna {wallet} number bhejo, kal wale khane ke paise bhejne hain"),
    ("mobile_wallet_phishing", "NORMAL", "legit", "Urdu", "{wallet} سے {amt} روپے موصول ہو گئے، شکریہ"),
    ("mobile_wallet_phishing", "NORMAL", "legit", "English", "Mobile load of Rs 100 successful via {wallet}"),
    ("mobile_wallet_phishing", "NORMAL", "hardneg", "Roman-Urdu", "{wallet} kabhi bhi call karke MPIN nahi poochta, apna PIN kisi ko mat batayein"),
    ("mobile_wallet_phishing", "NORMAL", "hardneg", "English", "Your {wallet} verification code is {code}. Never share it, not even with our staff"),
    ("mobile_wallet_phishing", "NORMAL", "hardneg", "Roman-Urdu", "Agar koi bole galti se paise chale gaye wapas bhejo, to pehle apni app mein balance check karo"),
    ("mobile_wallet_phishing", "NORMAL", "hardneg", "Code-Mixed", "Lucky draw ya inaam wale messages par apna PIN mat dena, ye sab fraud hota hai"),
    ("mobile_wallet_phishing", "NORMAL", "hardneg", "Urdu", "اپنا پن کوڈ کسی کو نہ بتائیں، کمپنی کبھی آپ سے پن نہیں مانگتی"),

    # ---------------- Job / online earning ----------------
    ("job_fraud", "SCAM", "scam", "Roman-Urdu", "Ghar baithe roz {amt} kamayein, sirf 1000 registration fee bhejein aur kaam shuru karein"),
    ("job_fraud", "SCAM", "scam", "English", "Congratulations! You are selected for a remote data-entry job. Pay Rs {amt} security deposit to start"),
    ("job_fraud", "SCAM", "scam", "Code-Mixed", "YouTube videos like karo aur daily {amt} earn karo, pehle task ke liye 500 invest karna hoga"),
    ("job_fraud", "SCAM", "scam", "Roman-Urdu", "Dubai mein job pakki hai, visa process ke liye {amt} aaj hi jama karwain, seats limited hain"),
    ("job_fraud", "SCAM", "scam", "Urdu", "گھر بیٹھے روزانہ {amt} روپے کمائیں، رجسٹریشن فیس بھیجیں اور کام شروع کریں"),
    ("job_fraud", "SCAM", "scam", "Roman-Urdu", "Aapka CV select ho gaya hai, interview se pehle training fee {amt} {wallet} karein"),
    ("job_fraud", "NORMAL", "legit", "English", "Thank you for applying. Your interview is scheduled for {time} at our {city} office"),
    ("job_fraud", "NORMAL", "legit", "Roman-Urdu", "{name} tumhara interview kaisa gaya? Result kab tak ayega"),
    ("job_fraud", "NORMAL", "legit", "Code-Mixed", "Offer letter email kar diya hai, joining next month se hai, documents le aana"),
    ("job_fraud", "NORMAL", "legit", "Roman-Urdu", "Office {time} tak band hai, report kal subah submit kar dena"),
    ("job_fraud", "NORMAL", "legit", "Urdu", "آپ کا انٹرویو {time} ہے، اصل دستاویزات ساتھ لائیں"),
    ("job_fraud", "NORMAL", "hardneg", "English", "Genuine employers never ask for a registration or security fee. Do not pay anyone to get a job"),
    ("job_fraud", "NORMAL", "hardneg", "Roman-Urdu", "Job ke liye agar koi pehle fee maange to samajh jao fraud hai, hum koi fee nahi lete"),
    ("job_fraud", "NORMAL", "hardneg", "Code-Mixed", "Ghar baithe kamai wale ads se bacho, pehle company verify karo phir apply karo"),

    # ---------------- Everyday chats (no scam topic) ----------------
    ("none", "NORMAL", "legit", "Roman-Urdu", "{name} kal class {time} hai, notes le aana"),
    ("none", "NORMAL", "legit", "Roman-Urdu", "Chai pe chalein? Main {time} free hun"),
    ("none", "NORMAL", "legit", "Roman-Urdu", "{rel} sabzi le aaun? Tamatar aur pyaz khatam hain"),
    ("none", "NORMAL", "legit", "Roman-Urdu", "Match dekha kal? Last over mein kya kamaal ki batting thi"),
    ("none", "NORMAL", "legit", "Roman-Urdu", "Shaadi ka card aa gaya hai, {city} mein hai, sab chalenge"),
    ("none", "NORMAL", "legit", "Roman-Urdu", "Light gayi hui hai, UPS bhi khatam, {time} tak aa jayegi shayad"),
    ("none", "NORMAL", "legit", "Roman-Urdu", "Jummah ki namaz ke baad mil lena, kuch zaroori baat karni hai"),
    ("none", "NORMAL", "legit", "Code-Mixed", "Assignment submit kar diya? Deadline {time} hai"),
    ("none", "NORMAL", "legit", "Code-Mixed", "Traffic bohat hai, 15 minute late ho jaunga, sorry"),
    ("none", "NORMAL", "legit", "Code-Mixed", "{name} weekend pe movie ka plan hai, tum aaoge?"),
    ("none", "NORMAL", "legit", "English", "Happy birthday {name}! Have an amazing year ahead"),
    ("none", "NORMAL", "legit", "English", "Can you share the photos from yesterday's dinner?"),
    ("none", "NORMAL", "legit", "Urdu", "کل {time} ملتے ہیں، کتاب ساتھ لے آنا"),
    ("none", "NORMAL", "legit", "Urdu", "عید مبارک! سب گھر والوں کو سلام کہنا"),
    ("none", "NORMAL", "legit", "Roman-Urdu", "Eid mubarak {rel}, sab ko salam kehna, InshaAllah jald milenge"),
]


def fill(template: str, rng: random.Random) -> str:
    out = template
    for key, options in SLOTS.items():
        token = "{" + key + "}"
        if token in out:
            out = out.replace(token, rng.choice(options))
    return out


def build() -> pd.DataFrame:
    rng = random.Random(SEED)
    rows, next_id = [], FIRST_ID
    for t_idx, (theme, label, kind, language, template) in enumerate(T):
        seen = set()
        for _ in range(VARIANTS * 10):         
            text = fill(template, rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append({
                "id": next_id,
                "group_id": f"v2_{t_idx:03d}",
                "text": text,
                "label": label,
                "language": language,
                "scam_type": theme,
                "source": SOURCE,
                "synthetic": True,
                "v2_kind": kind,
            })
            next_id += 1
            if len(seen) == VARIANTS:
                break
    return pd.DataFrame(rows)


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, lineterminator="\n")
    print(f"[make_v2] wrote {len(df)} rows to {OUT}")
    print(df.groupby(["v2_kind", "label"]).size().to_string())
    print(df["language"].value_counts().to_string())


if __name__ == "__main__":
    main()
