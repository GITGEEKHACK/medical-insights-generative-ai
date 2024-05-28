import os

USER_DATA_PATH = os.getenv("USER_DATA_PATH")


class BotoClient:
    AWS_DEFAULT_REGION = "us-east-1"
    read_timeout = 3600
    connect_timeout = 3600


class AllowedFileType:
    IMAGE = ['jpg', 'png', 'jpeg']
    PDF = ['pdf']


class ExceptionMessage:
    INVALID_FILE_EXCEPTION_MESSAGE = 'Input file is invalid'
    FAILED_TO_DOWNLOAD_FILE_FROM_URL_EXCEPTION_MESSAGE = 'Unable to download file from url'
    NO_IMAGE_FOUND_EXCEPTION_MESSAGE = "Unable to find image in PDF"
    FILE_LIMIT_EXCEEDED_EXCEPTION_MESSAGE = "File limit exceeded, only upload file upto 25 MB"
    MAX_FILE_SIZE_EXCEEDED_ERROR = "File limit exceeded, only upload file upto 100 MB"
    FILE_PATH_NULL_EXCEPTION_MESSAGE = "File is not uploaded"
    INPUT_QUERY_NULL_EXCEPTION_MESSAGE = "Input query is empty or null"
    MULTIPLE_FILE_UPLOADED_EXCEPTION_MESSAGE = "Multiple files are uploaded, only upload single file"
    MISSING_REQUEST_BODY_EXCEPTION_MESSAGE = "Missing request body in the request"
    INVALID_REQUEST_BODY_EXCEPTION_MESSAGE = "Invalid request body received"
    FILE_UPLOAD_LIMIT_REACHED_EXCEPTION_MESSAGE = "Uploaded more than {x} files"
    TOTAL_PAGE_EXCEEDED_EXCEPTION_MESSAGE = "Combined document pages must not exceed {page_count_threshold}. Please upload fewer pages."
    FOLDER_PATH_NULL_EXCEPTION_MESSAGE = "Project not uploaded"
    MISSING_RESPONSE_LIST_EXCEPTION_MESSAGE = "Missing Response List Error"


class MedicalInsights:
    TOTAL_PAGES_THRESHOLD = 1000
    REQUEST_FOLDER_NAME = "request"
    RESPONSE_FOLDER_NAME = "response"

    class Prompts:
        PROMPT_TEMPLATE = """
        Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, don't try to make up an answer.
        <context>
        {context}
        </context>

        Question: {question}

        Assistant:
        """

        DOC_TYPE_PROMPT = """
        Using the information provided within the medical text excerpt, analyze the key elements such as the chief complaint, medical history, physical examination findings, procedures, and treatments mentioned. Based on these elements, determine the medical specialty or context that the document pertains to.
        If the document type is not immediately clear from a known category, provide your best inference based on the content.

        Strictly respond with the classification of the document in JSON format, adhering to the following structure:
        {"document_type": "Identified_Document_Type"}
        
        If the document type does not match any known categories, respond with:
        {"document_type": "Other"}
        
        Ensure that the document type is identified accurately based on the key elements within the text, even in the absence of an explicit title or heading.
        
        """

        MEDICAL_CHRONOLOGY_PROMPT = """
        The above text is obtained from medical records. Based on the information provided, you are tasked with extracting the 'Encounter Date' and corresponding 'Event' along with 'Doctor', 'Institution', and 'Reference' from medical records.

        'Encounter Date': In medical records, it is defined as the specific date when a patient had an interaction with a healthcare provider. This could be a visit to a clinic, a hospital admission, a telemedicine consultation, or any other form of medical service.
        Notes to keep in mind while extracting 'Encounter Date' :
        - Ensure 'Encounter Date' should also include other types of Protected Health Information dates which are listed below :
          1. 'Injury date': In medical records, it is defined as the specific date when a patient sustained any type of injury.
          2. 'Admission date': In medical records, it is defined as the specific date when a patient is officially admitted to a healthcare facility, such as a hospital or clinic, for inpatient care.
          3. 'Discharge date': In medical records, it is defined as the specific date when a patient is discharged or released from a healthcare facility after receiving treatment or completing a course of care.
        - Ensure 'Encounter Date' should also include other types of dates which are listed below :
          1. 'E-Signature date': In medical records, it is defined as the specific date when the record is electronically signed by or verified/reviewed by the practitioner having a professional designation.
          2. 'Expected date': In medical records, it is defined as the specific expected date of delivery and AUA.
        - Avoid giving any other types of dates like 'Birth date', 'Received date', 'Printed date', and 'Resulted date' :
          1. 'Birth date': In medical records, it is defined as the specific date when a patient is born. It is typically recorded and used for identification, legal, and administrative purposes. It is also used to calculate a person's age.
          2. 'Received date': In medical records, it is defined as the specific date when a lab, hospital, or clinic received the test result.
          3. 'Printed date': In medical records, it is defined as the specific date when the document was created, updated, or reviewed.
          4. 'Resulted date': In medical records, it is defined as the specific date when the results of certain tests, procedures, or treatments are made available or reported.
        - Ensure all the actual 'Encounter Date' are strictly converted to the same format of 'MM/DD/YYYY'.
        - Strictly provide all the actual 'Encounter Date' starting from oldest to newest. Ensure none of the actual 'Encounter Date' is left behind. Ensure dates from Past Medical History / Past Surgical History are also included.

        'Event': It is associated with the corresponding 'Encounter Date'. It is described as the summary of all activities that occurred on that particular 'Encounter Date'.
        Notes to keep in mind while extracting 'Event' :
        - Ensure all 'Event' descriptions should include the key points, context, and any relevant supporting details.
        - Also ensure all 'Event' descriptions are more detailed, thorough, and comprehensive yet a concise summary in medium-sized paragraphs.

        'Doctor': In medical records, a 'Doctor' is typically defined as the name of a licensed medical professional. A 'Doctor' is responsible for attending, diagnosing, and treating illnesses, injuries, and other health conditions. A 'Doctor' often provides preventive care, and health education to patients during the encounter.
        Notes to keep in mind while extracting 'Doctor' :
        - Ensure that the 'Doctor' pertains to the relevant 'Encounter Date' and 'Event'.
        - In case, if 'Doctor' is not found then find the name of 'Doctor' working at or related to the specific place which is most relevant to the extracted 'Encounter Date' and 'Event'.
        - In case, if 'Doctor' is still not found then keep the value of 'Doctor' as "None".

        'Institution': In medical records, it is defined as the specific workplace of a 'Doctor'. It should also include the name of the hospital.
        Notes to keep in mind while extracting 'Institution' :
        - Ensure that the 'Institution' pertains to the relevant 'Encounter Date', 'Event', and 'Doctor'.
        - In case, if 'Institution' is not found then keep the value of 'Institution' as "None".

        'Reference': It is an exact reference text from the medical record that is most relevant to the extracted 'Encounter Date' and 'Event'.
        Notes to keep in mind while extracting 'Reference' :
        - Ensure to provide the exact reference text avoiding any deviations from the original text.
        - Strictly ensure to restrict the length of 'Reference' to the medium-sized phrase.

        Note: This extraction process is crucial for various aspects of healthcare, including patient care tracking, scheduling follow-up appointments, billing, and medical research. Your attention to detail and accuracy in this task is greatly appreciated.

        Please ensure that you strictly provide a result in the below specified format of a list of tuples that includes 'Encounter Date', 'Event', 'Doctor', 'Institution', and 'Reference' :
        [ ("Encounter Date", "Event", "Doctor", "Institution", "Reference") ]
        """

        DIAGNOSIS_TREATMENT_ENTITY_PROMPT = """
        Task: Identify diagnoses and treatments from provided text without extra information.

        The definition of a valid diagnosis, valid treatment and valid medications is given below:
        Diagnosis: It is a process of identifying a patient's medical condition based on the evaluation of symptoms, history, and clinical evidence.
        Treatment: A medical intervention or series of actions conducted by healthcare professionals aimed at alleviating, managing, or curing symptoms and diseases, as well as improving a patient's overall health and well-being. This can include medication, surgery, therapy, lifestyle modifications, and other forms of care.
        PMH (Past Medical History): It is a record of a patient's health information regarding previous illnesses, surgeries, injuries, treatments, and other relevant medical events in life.

        Guidelines for Extraction:
        1. Exclude negated diagnosis and treatment information.
        2. Categorize diagnoses as allergy, PMH, or current condition.
        3. If an allergy is mentioned in the context of Past Medical History, it should be STRICTLY included in both the Allergy and PMH categories to reflect its ongoing relevance to the patient's medical history.
        4. Include signs, injuries, chronic pain, and medical conditions as diagnoses.
        5. Avoid repetition of PMH, allergies, and clear diagnoses.
        6. Extract only therapeutic procedures, surgeries, or interventions as treatments, including manual and physical therapies, as well as recommended exercises. Exclude past medical procedures that are not therapeutic interventions.
        7. Don't include specific medication names and dosages in treatments.
        8. Avoid misinterpreting symptoms as diagnoses or treatments.
        9. Include system organs and direction of organs with medical entities.
        10. Exclude hypothetical and conditional statements.
        11. Categorize entities under appropriate PMH based on identification, ensuring treatments are not included in diagnosis and vice versa.
        12. Exclude general advice or non-specific interventions from the treatments category.
        13. Exclude diagnostic procedures such as blocks or imaging tests from the treatments category.

        Output Response Format:
        1. Clear distinction between diagnosis and treatment entities.
        2. Exclude tests from diagnoses.
        3. Exclude medication from treatments.
        4. Exclude clinical findings, physical exam findings, and observations.
        5. Exclude doctor and patient names.
        6. Avoid repeating diagnoses and treatments if referring to the same condition or treatment.

        Please provide a JSON response strictly using the format below. Use this response as an example, but do not include the entity if it is not present, and include empty string values for missing keys:
        {
          "diagnosis":{
            "allergies":["Peanuts"],
            "pmh":["Type 2 Diabetes"],
            "others":["Hypertension"]
          },
          "treatments":{
            "pmh":["NORCO"],
            "others":["REST"]
          }
        }
        """

        TESTS_MEDICATION_ENTITY_PROMPT = """
        Your task is to identify valid procedures and valid medications from the user-provided text without including additional information, notes, and context.
        
        The definitions of the medical terms are given below:
        PMH (Past Medical History): It is a record of a patient's health information regarding previous illnesses, surgeries, injuries, treatments, and other relevant events in life.
        Treatment: It is a proven, safe, and effective therapeutic intervention aligned with medical standards, to manage or cure a diagnosed health condition.
        Tests: It refers to a comprehensive array of standardized methods and procedures designed to assess a patient's health. These tests cover the evaluation of the musculoskeletal system's integrity, neurological function, and overall physical condition. They include clinical assessments such as hands-on physical techniques—like palpation and range of motion exercises—and specialized tests that elicit responses indicative of medical conditions, including orthopedic and neurological examinations. Furthermore, tests incorporate laboratory analyses that investigate biological samples to diagnose diseases, monitor the course of an illness, and evaluate the success of treatments.
        Medications: It refers to drugs or substances used to treat, prevent, or diagnose diseases, relieve symptoms, or improve health.
        Dosage: It refers to the specific amount of a drug to be taken at one time or within a certain period, as prescribed by a healthcare professional.
        
        Please follow the below guidelines:
        1. Identify and categorize medications as past medical history (PMH) or current medications, including dosage.
        2. Exclude 'Medication' entity from the medication category.
        3. Extract all procedures from the document, ensuring each is linked with the relevant date.
        4. Include both tests and lab tests under the procedure category without differentiating between them.
        5. Ensure treatments are not mistaken as medications.
        6. Avoid repeating entities across all categories and sub-categories.
        7. Don't consider vitals information such as Height, weight, temperature, Blood pressure etc as Tests or Lab Tests.
        8. Don't include therapeutic procedures and treatments in Tests and Lab Tests.
        
        Please provide a JSON response strictly using the format below. Strictly use this response as an example. Strictly include empty string values for missing keys:
        {
          "tests":[
            {
              "date":"2023-03-25",
              "name":"Electrocardiogram (ECG)"
            },
            {
              "date":"2023-03-30",
              "name":"Hemoglobin A1c"
            },
            {
              "date":"2023-03-31",
              "name":"MRI scan of the brain"
            }
          ],
          "medications":{
            "pmh":[
              {
                "name":"Metformin",
                "dosage":"500 mg twice daily"
              }
            ],
            "others":[
              {
                "name":"Lisinopril",
                "dosage":"10 mg once daily"
              }
            ]
          }
        } 
        """

        SUMMARY_PROMPT = """Generate a detailed and accurate summary based on the user's input, concentrating specifically on identifying key medical diagnoses, outlining treatment plans, and highlighting pertinent aspects of the patient's medical history. Ensure precision and conciseness to deliver a focused and insightful summary, including the patient's name, age, and hospital name if provided. Avoid any suggestions or misconceptions not presented in the document."""

        CONCATENATE_SUMMARY = "Concatenate the summaries and remove the duplicate information from the summaries and make one summary without losing any information."

        HISTORY_PROMPT = """
        Model Instructions:
        Social History:
        Smoking: Check if there is any history of smoking (current or past). If there is, respond with "Yes" for both "Smoking" and "Tobacco". If there is no smoking history, proceed to evaluate tobacco use.
        Alcohol: Determine if the patient has ever consumed alcohol. If the patient currently consumes alcohol or has in the past, respond with "Yes". If not, respond with "No".
        Tobacco: If there is no history of smoking, assess the use of smokeless tobacco products such as chewing tobacco or snuff. Respond with "Yes" for "Tobacco" only if such non-smoking tobacco use is present. If there is no use of any tobacco products, respond with "No" for both "Smoking" and "Tobacco".

        Family History:
        Additional Information: Do not generate information beyond what is provided.
        Medical vs. Family History: Do not confuse the patient's medical history with their family history.
        Condition Details: List only the names of medical conditions, omitting details like onset, age, or timing.
        Key-Value Pairs: Use key-value pairs to represent significant medical histories of family members. The key is the family member's relation to the patient, and the value is a concise description of their significant medical history.
        Unspecified Relations: If a relation is not specified, use "NotMentioned" as the key. Do not include the "NotMentioned" key if there is no significant history.
        Exclusions: Leave out family members without a significant medical history, non-medical information, and personal identifiers. Focus solely on health conditions relevant to the patient's medical or genetic predisposition. Also exclude outputs such as {'NotMentioned': 'None'} because it is irrelevant for usecase. 

        JSON Template:
        Fill in the JSON template with the appropriate responses based on the medical text provided. Ensure that only family members with significant medical histories are included in the "Family_History" section.
        {
          "Social_History": {
            "Smoking": "Yes or No",
            "Alcohol": "Yes or No",
            "Tobacco": "Yes or No"
          },
          "Family_History": {
            // Insert key-value pairs for family members with significant medical history
            // Omit entries for family members without significant history
          }
        }
        """

        PSYCHIATRIC_INJURY_PROMPT = """
        Instructions for the model:
        Psychiatric Injury:       
        Compile a list of the names of psychiatric injuries or disorders the patient may have. This should encompass any diagnosed mental illnesses, traumatic brain injuries, psychological traumas, or other psychiatric conditions.
        Provide only the names of the psychiatric injuries or disorders without any additional details such as onset, treatment, or management strategies.

        Extract the relevant information from the provided medical text regarding the patient's psychiatric injuries or disorders and record your findings in the JSON template provided below:
        {
          "Psychiatric_Injury": ["name of injury or disorder", "another injury or disorder", ...]
        }
        """

        PATIENT_DEMOGRAPHICS_PROMPT = """
        Your task is to identify the name, date of birth, age, gender, height and weight from the user-provided text without including additional information, notes, and context.

        Please follow the below guidelines:
        1) Consider Date of Birth, Birth Date, and DOB as date_of_birth.
        2) Use Age or years old to identify the patient's age. Do not derive age from the date_of_birth. If multiple age values are mentioned, choose the highest value.
        3) Consider Patient Name, only Patient, only Name, and RE as patient_name.
        4) Use Gender, Sex, Male, Female, M, F, Mr. , Mrs. to identify the patient's gender. Do not assume or derive gender from the patient_name.
        5) Use Height, Ht to identify the patient's height. Strictly provide the height in inches only. If the height value of patient is present multiple time, ensure to return the current height of the patient.
            - Also provide the recent date on which the measurements of height were taken.
        6) Use Weight, Wt to identify the patient's weight. Strictly provide the weight in pounds only. If the weight value of patient is present multiple time, ensure to return the current weight of the patient.
            - Also provide the recent date on which the measurements of height were taken.
        7) The `date` for both `height` and `weight` should be the most recent date on which these measurements were taken.
        8) Don't provide your analysis in the final response.
        9) If the weight is present in kilograms(kgs), convert it into pounds(lbs).
        10) If the height is present in centimeters(cm), convert it into inches(in).

        Please strictly only provide a JSON result as given below:
        {
          "patient_name": "",
          "date_of_birth": "",
          "age": "",
          "gender": "",
          "height": {
            "value": "",
            "date": ""
          },
          "weight": {
            "value": "",
            "date": ""
          }
        }

        Ensure that height and weight are accurately reflected according to the specified units of measurement.
        Note: If any of the value is not found, fill the value with an empty string.
        """

    class TemplateResponse:
        DEMOGRAPHICS_TEMPLATE_RESPONSE = {"patient_name": "", "date_of_birth": "", "age": "", "gender": "",
                            "height": {"value": "", "date": ""}, "weight": {"value": "", "date": ""}, "bmi": ""}

        HISTORY_TEMPLATE_RESPONSE = {'social_history': {'page_no': None, 'values': {'Smoking': 'No', 'Alcohol': 'No', 'Tobacco': 'No'}}, 'family_history': {'page_no': None, 'values': {}}, 'psychiatric_injury': {'page_no': None, 'values': []}}

        QNA_EMPTY_DOC_RESPONSE = {'query': "", 'result': "It seems that the PDF document is empty", 'source_documents': []}
    class LineRemove:
        SUMMARY_FIRST_LAST_LINE_REMOVE = [
            'Based on the provided',
            'Here is a detailed',
            'Here is a consolidated',
            'Based on the',
            'Based on the detailed',
            'Based on the provided',
            'Based on the information provided',
            'Here is the',
            'In summary,']

    class RegExpression:
        """
        The DATE_EVENT_DOCTOR_INSTITUTION_REFERENCE regex pattern matches the date, event, doctor, institution, and reference
        Example:
            str = "[("01/11/2013", "event-1", "Mr. abc", "inst-1", "ref-1"),
                      ("05/17/2012","event-2","Mr. xyz","inst-2","ref-2")]"

            matches = re.findall(r'\((\"[\d\/]+\")\s*,\s*\"([^\"]+)\"\s*,\s*\"([^\"]+)\"\s*,\s*\"([^\"]+)\"\s*,\s*\"([^\"]+)\"', str)

            matches = [('"01/11/2013"', 'event-1', 'Mr. abc', 'inst-1', 'ref-1'),
                        ('"05/17/2012"', 'event-2', 'Mr. xyz', 'inst-2', 'ref-2')]

            If no matches are found, an empty list is returned:
            matches = []


        The DATE regex pattern matches the date for variable lengths
        Example:
            str = "10-20-2016 to 16-2020"

            matches = re.findall(r'(?:\d{1,2}-\d{1,2}-\d{1,4})|(?:\d{1,2}-\d{1,4})|(?:\d{1,4})', str)

            matches = ['10-20-2016', '16-2020']

            If no matches are found, an empty list is returned:
            matches = []
        """
        DATE_EVENT_DOCTOR_INSTITUTION_REFERENCE = r'\((\"[\d\/]+\")\s*,\s*\"([^\"]+)\"\s*,\s*\"([^\"]+)\"\s*,\s*\"([^\"]+)\"\s*,\s*\"([^\"]+)\"'
        DATE = r'(?:\d{1,2}-\d{1,2}-\d{1,4})|(?:\d{1,2}-\d{1,4})|(?:\d{1,4})'