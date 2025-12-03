import streamlit as st
import pandas as pd
import io
import os
from google import genai
from google.genai import types

# --- 1. CONFIGURATION AND UTILITIES ---

# Schema ที่เรากำหนดให้ LLM ต้องสกัด
GEO_SCHEMA = [
    "ID_EVENT",
    "LOCATION_NAME",
    "LATITUDE_EST",
    "LONGITUDE_EST",
    "EVENT_TYPE",
    "MAGNITUDE_SIZE",
    "DAMAGE_SUMMARY"
]
SCHEMA_HEADERS = ",".join(GEO_SCHEMA)

def generate_geo_csv(client: genai.Client, text_input: str, headers: str) -> str:
    """
    เรียกใช้ Gemini API เพื่อสกัดข้อมูลและบังคับให้สร้าง CSV
    
    Args:
        client: genai.Client instance.
        text_input: ข้อความบรรยายเชิงภูมิศาสตร์จากผู้ใช้
        headers: ชื่อคอลัมน์ CSV ที่ต้องการ

    Returns:
        String ที่เป็น CSV Output จาก LLM
    """
    
    # Prompt ที่มีความซับซ้อนและเน้นการบังคับโครงสร้าง (Structured Output Constraint)
    system_prompt = (
        "You are an expert Geospatial Data Generator. Your task is to analyze "
        "the provided narrative text and extract key geospatial and event-related "
        "information. Crucially, you **must** output the results **only** as a "
        "valid CSV format, including the header row. "
        "For LATITUDE_EST and LONGITUDE_EST, provide the estimated decimal coordinates "
        "based on the most specific location mentioned in the text. "
        "Do not include any introductory or concluding text, explanations, or code fences."
    )
    
    # สร้าง Prompt สำหรับ User
    user_prompt = f"""
    Analyze the following narrative text and extract the required information into a CSV format.
    
    REQUIRED CSV HEADERS (Schema):
    {headers}
    
    NARRATIVE TEXT TO ANALYZE:
    ---
    {text_input}
    ---
    
    Your output MUST start with the header row and contain ONLY the CSV data.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                # ตั้งค่าอุณหภูมิให้ต่ำเพื่อเพิ่มความน่าเชื่อถือในการสร้างโครงสร้างที่แม่นยำ
                temperature=0.1 
            )
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return ""

# --- 2. STREAMLIT APP LAYOUT ---

st.set_page_config(
    page_title="Structured Geo-Data Generator (SG-DG) 🌍",
    layout="wide"
)

st.title("🌍 Structured Geo-Data Generator (SG-DG)")
st.caption("แปลงข้อความบรรยายเชิงภูมิศาสตร์ให้เป็นตาราง CSV ที่พร้อมใช้งานด้วย Gemini API")

# --- Sidebar for API Key and Schema ---

with st.sidebar:
    st.header("🔑 Gemini API Configuration")
    
    # 1. API Key Input
    # ให้ผู้ใช้กรอก API Key หรือใช้จาก Streamlit Secret/Environment Variable
    api_key = st.text_input(
        "Enter your Gemini API Key", 
        type="password", 
        help="หาได้จาก Google AI Studio"
    )
    
    st.divider()
    
    # 2. Schema Preview
    st.header("📋 Target CSV Schema")
    st.info("LLM จะสกัดข้อมูลตามคอลัมน์เหล่านี้:")
    for header in GEO_SCHEMA:
        st.markdown(f"- **{header}**")

# --- Main Content ---

if not api_key:
    st.warning("โปรดใส่ **Gemini API Key** ใน Sidebar เพื่อเริ่มต้นใช้งาน")
else:
    try:
        # Initialise the Gemini Client
        # เราจะสร้าง client ภายใน else block เพื่อให้แน่ใจว่ามี API Key
        client = genai.Client(api_key=api_key)
        
        # 3. Text Input from User
        st.header("1. ป้อนข้อความบรรยายเชิงภูมิศาสตร์")
        geo_text_input = st.text_area(
            "คัดลอกและวางรายงานข่าว, บันทึกเหตุการณ์, หรือข้อมูลสำรวจภาคสนามที่นี่:",
            height=200,
            value="วันที่ 25 มกราคม 2568 เกิดเหตุน้ำท่วมหนักบริเวณถนนสุขุมวิท ในเขตเมืองพัทยา จังหวัดชลบุรี คาดว่าความเสียหายต่อทรัพย์สินอยู่ที่ประมาณ 50 ล้านบาท ระดับน้ำสูงสุดวัดได้ 80 เซนติเมตร"
        )
        
        # 4. Processing Button
        if st.button("🚀 สกัดและแปลงเป็น CSV", type="primary"):
            if geo_text_input:
                with st.spinner("กำลังวิเคราะห์และสร้าง CSV ด้วย Gemini API..."):
                    # เรียกใช้ฟังก์ชันสกัดข้อมูล
                    csv_output_text = generate_geo_csv(client, geo_text_input, SCHEMA_HEADERS)
                    
                if csv_output_text:
                    st.header("2. ผลลัพธ์ CSV ที่มีโครงสร้าง")
                    
                    try:
                        # ใช้ io.StringIO และ Pandas เพื่ออ่านข้อความ CSV ที่ LLM สร้างขึ้น
                        # และแปลงเป็น DataFrame
                        df_result = pd.read_csv(io.StringIO(csv_output_text))
                        
                        st.success("✅ สกัดข้อมูลและสร้าง DataFrame สำเร็จ!")
                        
                        # 5. Display Output (DataFrame)
                        st.dataframe(df_result, use_container_width=True)
                        
                        # 6. Download Feature
                        csv_download = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ ดาวน์โหลดผลลัพธ์เป็น CSV",
                            data=csv_download,
                            file_name='structured_geo_data.csv',
                            mime='text/csv',
                        )

                        st.subheader("Raw CSV Output (สำหรับตรวจสอบ)")
                        st.code(csv_output_text)
                        
                    except pd.errors.ParserError:
                        st.error("❌ Gemini API สร้าง CSV ที่ไม่ถูกต้องตามรูปแบบ (Parser Error) โปรดลองใหม่อีกครั้ง")
                        st.code(csv_output_text) # แสดง Raw Output เพื่อตรวจสอบ
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
                else:
                    st.warning("ไม่สามารถสร้าง CSV Output ได้ โปรดตรวจสอบ API Key และข้อความที่ป้อน")
            else:
                st.warning("โปรดป้อนข้อความบรรยายเพื่อทำการวิเคราะห์")
                
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการเริ่มต้น Client: {e}")