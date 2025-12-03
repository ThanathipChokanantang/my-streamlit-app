import streamlit as st
import pandas as pd
import io
import json
import plotly.express as px
from google import genai
from google.genai import types
from typing import List, Dict, Any, Tuple

# --- 1. FUNCTIONS FOR GEMINI INTERACTIONS ---
JSON_FORMAT_DESCRIPTION = """
[
  {
    "เวลา": "(เดือน/ปี หรือปีเท่านั้น, เช่น YYYY-MM หรือ YYYY)",
    "มูลค่าความเสียหาย(บาท)": (float หรือ number),
    "ผู้เสียชีวิต(คน)": (integer),
    "ผู้บาดเจ็บ(คน)": (integer),
    "แหล่งที่มา": "(ระบุชื่อสำนักข่าว/เว็บไซต์ **และ** URL ลิงก์อ้างอิง ถ้ามีการพยากรณ์ข้อมูล ให้เพิ่มข้อความต่อไปนี้ต่อท้าย: 'Gemini พยากรณ์ข้อมูลประเภท [มูลค่า_ความเสียหาย_บาท หรือ ผู้บาดเจ็บ_จำนวน] โดยอ้างอิงข้อมูลจาก [ระบุแหล่งข้อมูลที่ใช้ในการพยากรณ์]')"
    "รายละเอียดของเหตุการณ์": "(ข้อความสรุปเหตุการณ์ 100-300 คำ อธิบายสาเหตุ พื้นที่ และผลกระทบ **เป็นภาษาไทย**)"
  }
  // ... รายการเหตุการณ์อื่นๆ
]
"""

# กำหนดช่วงจำนวนเหตุการณ์ที่ต้องการ
MIN_EVENTS = 10
MAX_EVENTS = 100

# แปล Input เป็นภาษาอังกฤษ
def translate_to_english(client: genai.Client, text: str) -> str:
    """ใช้ Gemini เพื่อแปลข้อความภาษาไทยเป็นภาษาอังกฤษ"""
    
    st.info(f"...กำลังแปล '{text}' เป็นภาษาอังกฤษ...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[f"Translate the following Thai text to English: '{text}'"],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการแปลภาษา: {e}")
        return text

def create_raw_search_prompt_en(event_type_en: str, location_en: str) -> str:
    """Prompt สำหรับขั้นตอนที่ 1"""
    
    return (
        f"Search for historical statistics related to the disaster event type '{event_type_en}' that occurred in the region '{location_en}'. "
        "Focus on reports detailing the date/time, damage costs, number of fatalities, injuries, **clear news sources (website names/agency names) and their corresponding URLs**, and **brief event summaries**. "
        f"Summarize all findings into a **single, long text document** containing sufficient detail for subsequent statistical data extraction. Target between {MIN_EVENTS} and {MAX_EVENTS} separate historical events."
    )

def create_extraction_prompt(event_type_en: str, location_en: str) -> str:
    """Prompt สำหรับขั้นตอนที่ 2"""
    
    # System Prompt สำหรับการสกัดข้อมูล
    system_prompt = (
        "You are an expert in historical data analysis and estimation. Your task is to analyze the 'raw text' I provide, "
        "which summarizes disaster statistics for the event type '" + event_type_en + "' in '" + location_en + "', "
        "and extract the statistical data into a **100% correct JSON Array format**. "
        "Strict Rules: "
        f"1. The JSON Array must strictly adhere to this structure (with Thai keys):\n{JSON_FORMAT_DESCRIPTION}\n"
        "2. **IF DATA IS MISSING (มูลค่า_ความเสียหาย_บาท or ผู้บาดเจ็บ_จำนวน):** You MUST **predict/estimate** the value based on the other available data (e.g., fatalities, event scale, similar past events). The predicted value can be **0** if the context strongly suggests minimal or no impact, or if no reliable estimation can be made. "
        "3. **IF PREDICTION/ESTIMATION IS USED (including 0):** The 'แหล่งที่มา_ของ_ข่าว' column MUST include the original source information. Then, append a semicolon (;) followed by the specific prediction note in Thai: 'Gemini พยากรณ์ข้อมูลประเภท [ชื่อคอลัมน์ที่พยากรณ์] โดยอ้างอิงข้อมูลจาก [ระบุแหล่งข้อมูลที่ใช้ในการพยากรณ์ เช่น อัตราส่วนผู้เสียชีวิตต่อผู้บาดเจ็บ, การแปลงค่าเงิน, หรือขนาดความรุนแรงของภัยพิบัติ]'. If 0 is chosen as the prediction, state the reason clearly (e.g., 'ตั้งค่าเป็น 0 เนื่องจากขาดข้อมูลและเหตุการณ์มีความรุนแรงต่ำ')."
        "4. The 'แหล่งที่มา_ของ_ข่าว' column **MUST include the source name (e.g., BBC, NOAA) AND the specific URL/link** for the article, separated by a colon (e.g., 'Source Name: URL'). If multiple sources are used, separate them with a comma. "
        "5. The 'รายละเอียด_ของ_เหตุการณ์' column **MUST BE WRITTEN IN THAI** (100-300 words summary) based on the English source text. "
        "6. **HAVE AT LEAST " + str(MIN_EVENTS) + " EVENTS but NO MORE THAN " + str(MAX_EVENTS) + " EVENTS.**"
        "7. **NO TEXT** is allowed before or after the JSON Array."
    )
    return system_prompt

def get_raw_summary(client: genai.Client, event_type_en: str, location_en: str) -> str:
    """ขั้นตอนที่ 1: ใช้ Tool Search เพื่อค้นหาและสรุปข้อมูลดิบ"""
    
    prompt = create_raw_search_prompt_en(event_type_en, location_en)
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}]
            )
        )
        return response.text
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการค้นหาข้อมูล (ขั้นตอนที่ 1): {e}")
        return None

# --- 2. STREAMLIT APP LAYOUT ---

st.set_page_config(
    page_title="Disaster Event Statistics Analyzer",
    layout="wide"
)

st.title("Disaster Event Statistics Analyzer")
st.caption("Programming Assignment 4 วิชา 2209261 BASIG PROG NLP จัดทำโดย ธนาธิป โชคอนันตัง 6741005022 นิสิตภาควิชาภูมิศาสตร์ จุฬาลงกรณณ์มหาวิทยาลัย")

# --- 3. STREAMLIT APP LOGIC ---

# 🔑 Gemini API Configuration
with st.sidebar:
    st.header("Gemini API Input")
    api_key = st.text_input("Enter your Gemini API Key", type="password")


# 🔎 Main Content
if not api_key:
    st.warning("โปรดใส่ **Gemini API Key** ใน Sidebar เพื่อเริ่มต้นใช้งาน")
else:
    try:
        client = genai.Client(api_key=api_key)
        
        # 1.1 & 1.2: Input Event & Location
        st.header("ระบุเหตุการณ์และสถานที่")
        
        col1, col2 = st.columns(2)
        
        event_type_th = col1.text_input(
            "ประเภทภัยพิบัติ (เช่น น้ำท่วม, แผ่นดินไหว)",
            value="",
        )
        location_th = col2.text_input(
            "1.2 สถานที่ (เช่น หาดใหญ่, ญี่ปุ่น, ทั่วโลก)",
            value="",
        )
        
        # 2. Processing Button
        if st.button("ดำเนินการต่อ", type="primary"): 
            if event_type_th and location_th:
                
                # --- A. แปล Input เป็นภาษาอังกฤษ ---
                event_type_en = translate_to_english(client, event_type_th)
                location_en = translate_to_english(client, location_th)
                
                if not event_type_en or not location_en:
                    st.error("ไม่สามารถแปล Input ได้ โปรดลองใหม่อีกครั้ง")
                    st.stop()
                    
                st.success(f"แปลแล้ว: เหตุการณ์='{event_type_en}', สถานที่='{location_en}'")
                
                # --- ขั้นตอนที่ 1: ค้นหาข้อมูลดิบ (Tool Use) ---
                with st.spinner(f"...กำลังค้นหาข้อมูลเกี่ยวกับ '{event_type_en}' ใน '{location_en}'..."):
                    raw_text_summary = get_raw_summary(client, event_type_en, location_en)
                
                if not raw_text_summary:
                    st.stop()
                
                st.info("...ค้นพบข้อมูลดิบแล้ว กำลังเข้าสู่ขั้นตอนการสกัดข้อมูล...")

                # --- ขั้นตอนที่ 2: สกัดข้อมูล (Prompt Formatting) ---
                with st.spinner("...กำลังวิเคราะห์และสกัดข้อมูลเป็น JSON..."):
                    
                    system_prompt_extract = create_extraction_prompt(event_type_en, location_en)
                    user_prompt_extract = "Raw text to analyze:\n\n" + raw_text_summary

                    response_extract = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[user_prompt_extract],
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt_extract,
                        )
                    )
                    
                    json_output = response_extract.text.strip()
                    
                    # ทำความสะอาด JSON output ที่อาจมี ```json และ \n ติดมา
                    if json_output.startswith('```'):
                        json_output = json_output.strip().replace('```json', '').replace('```', '')

                if json_output:
                    
                    # --- ขั้นตอนที่ 3: แปลง JSON เป็น Pandas DataFrame และแสดงผล ---
                    st.header("ตารางสรุปสถิติเหตุการณ์")
                    
                    try:
                        data: List[Dict[str, Any]] = json.loads(json_output)
                        
                        num_events = len(data)

                        # --- ตรวจสอบจำนวนเหตุการณ์ ---
                        if num_events < MIN_EVENTS:
                            st.error(f"ข้อมูลไม่เพียงพอ! พบเหตุการณ์เพียง **{num_events}** ครั้ง (เป้าหมายคือ {MIN_EVENTS}-{MAX_EVENTS} ครั้ง) กรุณาลองระบุภัยพิบัติหรือสถานที่ที่กว้างขึ้น")
                            st.subheader("Raw JSON Output (สำหรับตรวจสอบ)")
                            st.code(json_output)
                            st.stop()
                        
                        if num_events > MAX_EVENTS:
                            st.error(f"ข้อมูลมากเกินไป! พบเหตุการณ์ถึง **{num_events}** ครั้ง (เป้าหมายคือ {MIN_EVENTS}-{MAX_EVENTS} ครั้ง) กรุณาลองระบุภัยพิบัติหรือสถานที่ที่แคบลง")
                            st.subheader("Raw JSON Output (สำหรับตรวจสอบ)")
                            st.code(json_output)
                            st.stop()

                        if num_events == 0:
                            st.error(f"ไม่พบข้อมูลสถิติเหตุการณ์ '{event_type_th}' ใน '{location_th}' ที่มีข้อมูลครบถ้วน")
                            st.warning("โปรดลองระบุสถานที่หรือประเภทภัยพิบัติที่กว้างขึ้น")
                            st.subheader("Raw JSON Output (สำหรับตรวจสอบ)")
                            st.code(json_output)
                            st.stop()
                            
                        df = pd.DataFrame(data)
                        
                        # ทำความสะอาดข้อมูลวันที่และเรียงลำดับ
                        df['เวลา_Sort'] = pd.to_datetime(df['เวลา'], errors='coerce')
                        df.dropna(subset=['เวลา_Sort'], inplace=True) 
                        df = df.sort_values(by='เวลา_Sort').reset_index(drop=True)
                        df.drop(columns=['เวลา_Sort'], inplace=True)
                        
                        st.success(f"สกัดและจัดโครงสร้างข้อมูลสำเร็จ! (จำนวนเหตุการณ์ที่สกัดได้: {len(df)} ครั้ง)")
                        
                        # กำหนดคอลัมน์ที่จะแสดงใน DataFrame (ไม่มี ลำดับ_เหตุการณ์)
                        display_cols = [
                            'เวลา',
                            'มูลค่าความเสียหาย(บาท)',
                            'ผู้เสียชีวิต(คน)',
                            'ผู้บาดเจ็บ(คน)',
                            'แหล่งที่มา',
                            'รายละเอียดของเหตุการณ์'
                        ]
                        
                        df_display = df.reindex(columns=display_cols, fill_value='')

                        st.dataframe(df_display, use_container_width=True)
                        
                        # --- ขั้นตอนที่ 4: สร้าง Bar Chart วิเคราะห์ ---
                        st.header("กราฟแสดงผลสถิติเหตุการณ์")
                        
                        # ใช้ Index ของ DataFrame แทนลำดับเหตุการณ์ในการสร้าง Label
                        df['ลำดับ_ชั่วคราว'] = df.index + 1
                        df['เหตุการณ์_Label'] = df['เวลา'] + ' (' + df['ลำดับ_ชั่วคราว'].astype(str) + ')'
                        
                        # 4.1 กราฟมูลค่าความเสียหาย
                        fig1 = px.bar(
                            df, 
                            x='เหตุการณ์_Label', 
                            y='มูลค่าความเสียหาย(บาท)', 
                            title='มูลค่าความเสียหาย (บาท) ในแต่ละเหตุการณ์',
                            labels={'มูลค่าความเสียหาย(บาท)': 'มูลค่าความเสียหาย (บาท)', 'เหตุการณ์_Label': 'เวลา'},
                            height=400,
                            color='มูลค่าความเสียหาย(บาท)',
                            color_continuous_scale=px.colors.sequential.Sunset
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # 4.2 กราฟจำนวนผู้เสียชีวิต
                        fig2 = px.bar(
                            df, 
                            x='เหตุการณ์_Label', 
                            y='ผู้เสียชีวิต(คน)', 
                            title='จำนวนผู้เสียชีวิต (คน) ในแต่ละเหตุการณ์',
                            labels={'ผู้เสียชีวิต(คน)': 'จำนวนผู้เสียชีวิต (คน)', 'เหตุการณ์_Label': 'เวลา'},
                            height=400,
                            color='ผู้เสียชีวิต(คน)',
                            color_continuous_scale=px.colors.sequential.Reds
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # 4.3 กราฟจำนวนผู้บาดเจ็บ
                        fig3 = px.bar(
                            df, 
                            x='เหตุการณ์_Label', 
                            y='ผู้บาดเจ็บ(คน)', 
                            title='จำนวนผู้ได้รับบาดเจ็บ (คน) ในแต่ละเหตุการณ์',
                            labels={'ผู้บาดเจ็บ(คน)': 'จำนวนผู้บาดเจ็บ (คน)', 'เหตุการณ์_Label': 'เวลา'},
                            height=400,
                            color='ผู้บาดเจ็บ(คน)',
                            color_continuous_scale=px.colors.sequential.Blues
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                        
                    except json.JSONDecodeError:
                        st.error("เกิดข้อผิดพลาดในการประมวลผล JSON")
                        st.warning("สาเหตุ: LLM ไม่สามารถสร้าง JSON ที่ถูกต้องตามโครงสร้างได้")
                        st.subheader("Raw JSON Output (สำหรับตรวจสอบ)")
                        st.code(json_output)
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูลหรือสร้างกราฟ: {e}")
                else:
                    st.warning("ไม่สามารถสร้าง Output จาก Gemini API ได้")
            else:
                st.warning("โปรดระบุประเภทภัยพิบัติและสถานที่")
                
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการเริ่มต้น Client: {e}")