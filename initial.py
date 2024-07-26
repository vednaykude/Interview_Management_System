import plotly.express as px
import streamlit as st
import whisper
import os
import tempfile
import sqlite3
from groq import Groq
from textblob import TextBlob
import pandas as pd
import re

# Initialize Groq client
client = Groq(api_key='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')

# Database setup
conn = sqlite3.connect('interview_assistant.db')
c = conn.cursor()

# Create tables if they do not exist
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL,
    role TEXT NOT NULL
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS user_data (
    username TEXT,
    question TEXT,
    response TEXT,
    feedback TEXT,
    relevance INTEGER,
    star_method INTEGER,
    specificity INTEGER,
    communication INTEGER,
    problem_solving INTEGER,
    professionalism INTEGER,
    adaptability INTEGER,
    total_score INTEGER,
    FOREIGN KEY(username) REFERENCES users(username)
)
''')

conn.commit()


# Define the scoring criteria
#def score_criteria(feedback, sentiment):
#    polarity, subjectivity = sentiment
 #   score = len(feedback.split()) + polarity * 10 - subjectivity * 5
 #   return max(0, score)  # Ensure score is non-negative


def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']


def generate_question(industry):
    prompt = (f"Generate a behavioral interview question for a candidate interviewing in the {industry} industry."
              f"Do not include the answer, just the question.")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    question = chat_completion.choices[0].message.content.strip()
    return question


scoring_criteria = """
Behavioral Interview Scoring Criteria

1. Relevance to Question (0-5 points)
   - 5 points: Answer is directly relevant and fully addresses the question.
   - 4 points: Answer is mostly relevant but misses some minor aspects.
   - 3 points: Answer is somewhat relevant but misses several key points.
   - 2 points: Answer is barely relevant and misses most key points.
   - 1 point: Answer is off-topic but related to the general theme.
   - 0 points: Answer is completely off-topic or no response given.

2. Use of STAR Method (0-5 points)
   - 5 points: Clear and complete use of Situation, Task, Action, and Result.
   - 4 points: Mostly clear use of STAR with minor details missing.
   - 3 points: Use of STAR but lacks clarity in one or two components.
   - 2 points: Attempted use of STAR but very unclear or incomplete.
   - 1 point: Minimal use of STAR elements.
   - 0 points: No use of STAR method.

3. Specificity and Detail (0-5 points)
   - 5 points: Provides detailed, specific examples with quantifiable results.
   - 4 points: Provides specific examples but lacks some detail or quantifiable results.
   - 3 points: Provides some detail but examples are somewhat vague or general.
   - 2 points: Provides minimal detail and very general examples.
   - 1 point: Provides very vague examples with no specific detail.
   - 0 points: No examples given or entirely non-specific response.

4. Communication Skills (0-5 points)
   - 5 points: Response is clear, concise, well-organized, and easy to understand.
   - 4 points: Response is mostly clear and well-organized with minor lapses.
   - 3 points: Response is understandable but has some clarity or organization issues.
   - 2 points: Response is somewhat difficult to follow due to clarity or organization issues.
   - 1 point: Response is mostly unclear or very poorly organized.
   - 0 points: Response is incomprehensible or no response given.

5. Problem-Solving and Critical Thinking (0-5 points)
   - 5 points: Demonstrates exceptional problem-solving skills and critical thinking.
   - 4 points: Demonstrates strong problem-solving skills and critical thinking.
   - 3 points: Demonstrates adequate problem-solving skills and critical thinking.
   - 2 points: Demonstrates limited problem-solving skills and critical thinking.
   - 1 point: Demonstrates very little problem-solving skills and critical thinking.
   - 0 points: Demonstrates no problem-solving skills or critical thinking.

6. Professionalism and Composure (0-5 points)
   - 5 points: Maintains complete professionalism and composure throughout.
   - 4 points: Mostly professional and composed with minor lapses.
   - 3 points: Generally professional but some noticeable lapses.
   - 2 points: Somewhat professional but several lapses in composure.
   - 1 point: Rarely professional with frequent lapses in composure.
   - 0 points: Not professional or composed at all.

7. Adaptability and Learning (0-5 points)
   - 5 points: Demonstrates excellent adaptability and willingness to learn.
   - 4 points: Demonstrates strong adaptability and willingness to learn.
   - 3 points: Demonstrates adequate adaptability and willingness to learn.
   - 2 points: Demonstrates limited adaptability and willingness to learn.
   - 1 point: Demonstrates very little adaptability and willingness to learn.
   - 0 points: Demonstrates no adaptability or willingness to learn.

Total Score: /35
"""


def analyze_response(question, response):
    prompt = (
            f"Question: {question}\nResponse: {response}\nAnalyze the response and provide feedback. Use 'you' in your response."
            f"use this rubric to grade the response, and put colons after the name of each criteria, it should"
            f"look like this: Problem-Solving and Critical Thinking (1 point). Here's the rubric:" + scoring_criteria)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    feedback = chat_completion.choices[0].message.content.strip()
    criteria_scores, total_score = extract_scores(feedback)
    return feedback, criteria_scores, total_score

def extract_scores(feedback):
    # Extract points for each criterion
    criteria_points = re.findall(r"\*\*(.*?) \((\d+) points?\)", feedback)
    criteria_dict = {criterion: int(points) for criterion, points in criteria_points}

    # Extract total score
    total_score_match = re.search(r"Total Score: (\d+)/(\d+)", feedback)
    if total_score_match:
        total_score = int(total_score_match.group(1))
    else:
        total_score = None

    return criteria_dict, total_score

def sentiment_analysis(response):
    blob = TextBlob(response)
    sentiment = blob.sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity
    sentiment_desc = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
    subjectivity_desc = "Subjective" if subjectivity > 0.5 else "Objective"
    return sentiment_desc, subjectivity_desc


def save_user_data(username, question, response, feedback, criteria_scores, total_score):
    c.execute('''
    INSERT INTO user_data (username, question, response, feedback, relevance, star_method, specificity,
                           communication, problem_solving, professionalism, adaptability, total_score)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (username, question, response, feedback,
          criteria_scores.get('Relevance to Question', 0),
          criteria_scores.get('Use of STAR Method', 0),
          criteria_scores.get('Specificity and Detail', 0),
          criteria_scores.get('Communication Skills', 0),
          criteria_scores.get('Problem-Solving and Critical Thinking', 0),
          criteria_scores.get('Professionalism and Composure', 0),
          criteria_scores.get('Adaptability and Learning', 0),
          total_score))
    conn.commit()


def load_user_data(username):
    c.execute('''
    SELECT username, question, response, feedback, relevance, star_method, specificity, communication, problem_solving, professionalism, adaptability, total_score FROM user_data WHERE username = ?
    ''', (username,))
    return c.fetchall()


def load_all_candidates():
    c.execute('''
    SELECT username FROM users WHERE role = "Interviewee"
    ''')
    return c.fetchall()


def clear_user_data(username):
    c.execute('''
    DELETE FROM user_data WHERE username = ?
    ''', (username,))
    conn.commit()


def filter_candidates_by_score(min_total_score=None, max_total_score=None, min_criteria_scores=None,
                               max_criteria_scores=None):
    query = '''
    SELECT 
        username,
        AVG(total_score) as average_score,
        AVG(relevance) as average_relevance,
        AVG(star_method) as average_star_method,
        AVG(specificity) as average_specificity,
        AVG(communication) as average_communication,
        AVG(problem_solving) as average_problem_solving,
        AVG(professionalism) as average_professionalism,
        AVG(adaptability) as average_adaptability
    FROM user_data
    GROUP BY username
    HAVING 1=1
    '''

    params = []

    if min_total_score is not None:
        query += ' AND AVG(total_score) >= ?'
        params.append(min_total_score)

    if max_total_score is not None:
        query += ' AND AVG(total_score) <= ?'
        params.append(max_total_score)

    for criteria, min_score in min_criteria_scores.items():
        query += f' AND AVG({criteria}) >= ?'
        params.append(min_score)

    for criteria, max_score in max_criteria_scores.items():
        query += f' AND AVG({criteria}) <= ?'
        params.append(max_score)

    c.execute(query, params)
    return c.fetchall()

def color_score(val):
    color = 'green' if val >= 4 else 'red'
    return f'background-color: {color}'

# Main Streamlit code
st.title("Interview Management System")

# Ensure session state is initialized
if 'role' not in st.session_state:
    st.session_state.role = None
if 'username' not in st.session_state:
    st.session_state.username = None

role = st.sidebar.selectbox("Select your role", ["Select", "Interviewee", "Interviewer"], index=0)

if role == "Select":
    st.sidebar.write("Please select a role to proceed.")

# Interviewee Login/Signup
if role == "Interviewee":
    if st.session_state.role != "Interviewee":
        st.sidebar.title("Interviewee Signup/Sign In")
        username = st.sidebar.text_input("Username", key="interviewee_username")
        password = st.sidebar.text_input("Password", type="password", key="interviewee_password")

        if st.sidebar.button("Signup", key="interviewee_signup"):
            if username and password:
                try:
                    c.execute('''
                    INSERT INTO users (username, password, role) VALUES (?, ?, ?)
                    ''', (username, password, "Interviewee"))
                    conn.commit()
                    st.session_state.role = "Interviewee"
                    st.session_state.username = username
                    st.sidebar.success("Signup successful! You are now logged in as an Interviewee.")
                except sqlite3.IntegrityError:
                    st.sidebar.error("Username already exists. Please choose a different username.")
            else:
                st.sidebar.error("Please enter a username and password.")

        if st.sidebar.button("Login", key="interviewee_login"):
            if username and password:
                c.execute('''
                SELECT password FROM users WHERE username = ?
                ''', (username,))
                user = c.fetchone()
                if user and user[0] == password:
                    st.session_state.role = "Interviewee"
                    st.session_state.username = username
                    st.sidebar.success("Login successful! Welcome back.")
                else:
                    st.sidebar.error("Incorrect username or password.")
            else:
                st.sidebar.error("Please enter a username and password.")

# Interviewee Flow
if st.session_state.role == "Interviewee":
    st.sidebar.write(f"Logged in as: {st.session_state.username}")

    tab1, tab2 = st.tabs(["Record Response", "View Profile"])

    # Record Response
    with tab1:
        st.header("Record Your Response")
        industry = st.text_input("Enter the industry you are interviewing for:")
        if st.button("Generate Question"):
            question = generate_question(industry)
            st.session_state.question = question
            st.write("Question:", question)

        uploaded_file = st.file_uploader("Upload your audio response", type=["mp3", "wav", "m4a"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                audio_path = temp_file.name
                transcription = transcribe_audio(audio_path)
                st.write("Transcription:", transcription)
                st.session_state.response = transcription

        if st.button("Submit Response"):
            if 'question' in st.session_state and 'response' in st.session_state:
                feedback, criteria_dict, total_score = analyze_response(st.session_state.question, st.session_state.response)
                save_user_data(st.session_state.username, st.session_state.question, st.session_state.response,
                               feedback, criteria_dict, total_score)
                st.write("Feedback:", feedback)
                st.write("Score:", total_score)
                st.success("Your response has been submitted and scored.")
            else:
                st.error("Please record your response and generate a question first.")

    # View Profile
    with tab2:
        st.header("View Your Profile")
        data = load_user_data(st.session_state.username)
        if data:
            df = pd.DataFrame(data, columns=['Username', 'Question', 'Response', 'Feedback', 'Relevance Score',
                                             'STAR Method Score',
                                             'Specificity Score', 'Communication Score',
                                             'Problem Solving Score', 'Professionalism Score',
                                             'Adaptability Score', 'Total Score'])

            st.write(df)
        else:
            st.write("No data available.")

    # Logout Button for Interviewee
    if st.button("Logout"):
        st.session_state.role = None
        st.session_state.username = None
        st.experimental_rerun()

# Interviewer Login/Signup
if role == "Interviewer":
    if st.session_state.role != "Interviewer":
        st.sidebar.title("Interviewer Signup/Sign In")
        username = st.sidebar.text_input("Username", key="interviewer_username")
        password = st.sidebar.text_input("Password", type="password", key="interviewer_password")

        if st.sidebar.button("Signup", key="interviewer_signup"):
            if username and password:
                try:
                    c.execute('''
                    INSERT INTO users (username, password, role) VALUES (?, ?, ?)
                    ''', (username, password, "Interviewer"))
                    conn.commit()
                    st.session_state.role = "Interviewer"
                    st.session_state.username = username
                    st.sidebar.success("Signup successful! You are now logged in as an Interviewer.")
                except sqlite3.IntegrityError:
                    st.sidebar.error("Username already exists. Please choose a different username.")
            else:
                st.sidebar.error("Please enter a username and password.")

        if st.sidebar.button("Login", key="interviewer_login"):
            if username and password:
                c.execute('''
                SELECT password FROM users WHERE username = ?
                ''', (username,))
                user = c.fetchone()
                if user and user[0] == password:
                    st.session_state.role = "Interviewer"
                    st.session_state.username = username
                    st.sidebar.success("Login successful! Welcome back.")
                else:
                    st.sidebar.error("Incorrect username or password.")
            else:
                st.sidebar.error("Please enter a username and password.")

# Interviewer Flow
if st.session_state.role == "Interviewer":
    st.sidebar.write(f"Logged in as: {st.session_state.username}")

    tab1, tab2 = st.tabs(["Dashboard", "Candidate Profiles"])

    # Candidate Profiles
    with tab2:
        st.header("Candidate Profiles")
        candidates = load_all_candidates()
        if candidates:
            candidate_names = [c[0] for c in candidates]
            selected_candidate = st.selectbox("Select a Candidate", candidate_names)
            if selected_candidate:
                data = load_user_data(selected_candidate)
                if data:
                    df = pd.DataFrame(data, columns=['Username', 'Question', 'Response', 'Feedback', 'Relevance Score',
                                             'STAR Method Score',
                                             'Specificity Score', 'Communication Score',
                                             'Problem Solving Score', 'Professionalism Score',
                                             'Adaptability Score', 'Total Score'])
                    st.write(f"Profile for {selected_candidate}:")
                    st.write(df)
                else:
                    st.write("No data available for the selected candidate.")

            # Clear Responses Button
            if st.button("Clear Responses for Selected Candidate"):
                if selected_candidate:
                    clear_user_data(selected_candidate)
                    st.success(f"All responses for {selected_candidate} have been cleared.")
                else:
                    st.error("No candidate selected.")
        else:
            st.write("No candidates available.")

        # Assuming this is part of your Streamlit app
        with tab1:
            st.header("Dashboard")
            st.write("You can access candidate profiles and their feedback here.")

            st.write("Filter candidates by score:")

            # Dropdown menu to select the criteria
            criteria_names = {
                'Total Score': None,  # Special case for total score
                'Relevance to Question': 'relevance',
                'Use of STAR Method': 'star_method',
                'Specificity and Detail': 'specificity',
                'Communication Skills': 'communication',
                'Problem-Solving and Critical Thinking': 'problem_solving',
                'Professionalism and Composure': 'professionalism',
                'Adaptability and Learning': 'adaptability'
            }

            selected_criteria = st.selectbox("Select Criteria to Filter On", list(criteria_names.keys()))

            if selected_criteria:
                if selected_criteria == 'Total Score':
                    min_total_score = st.slider("Minimum Total Score", 0, 35, 0)
                    max_total_score = st.slider("Maximum Total Score", 0, 35, 35)
                    min_criteria_scores = {}
                    max_criteria_scores = {}
                else:
                    criteria_column = criteria_names[selected_criteria]
                    min_criteria_scores = {criteria_column: st.slider(f"Minimum {selected_criteria} Score", 0, 5, 0,
                                                                      key=f"min_{criteria_column}")}
                    max_criteria_scores = {criteria_column: st.slider(f"Maximum {selected_criteria} Score", 0, 5, 5,
                                                                      key=f"max_{criteria_column}")}

                    # Set default for total score if only criteria other than total score is selected
                    min_total_score = 0
                    max_total_score = 35

                filtered_candidates = filter_candidates_by_score(
                    min_total_score=min_total_score,
                    max_total_score=max_total_score,
                    min_criteria_scores=min_criteria_scores,
                    max_criteria_scores=max_criteria_scores
                )

                def gradient_color(val, min_val, mid_val, max_val):
                    # Normalize the value to a scale of 0 to 1
                    if val >= mid_val:
                        norm_val = (val - mid_val) / (max_val - mid_val) * 0.5 + 0.5
                        color = f'rgba(0, 255, {int(255 * (1 - (norm_val - 0.5) * 2))}, 1)'  # Gradient from white to green
                    else:
                        norm_val = (mid_val - val) / (mid_val - min_val) * 0.5
                        color = f'rgba(255, {int(255 * (1 - norm_val))}, {int(255 * (1 - norm_val))}, 1)'  # Gradient from red to white

                    return f'background-color: {color}'


                def apply_gradient_color(df, column_name, min_val, mid_val, max_val):
                    return df.style.applymap(lambda val: gradient_color(val, min_val, mid_val, max_val),
                                             subset=[column_name])


                # Function to remove trailing zeroes from a float
                def remove_trailing_zeroes(num):
                    if '.' in str(num):
                        return float(str(num).rstrip('0').rstrip('.'))
                    else:
                        return num

                if filtered_candidates:
                    df = pd.DataFrame(filtered_candidates,
                                      columns=['Candidate', 'Average Score', 'Average Relevance', 'Average STAR Method',
                                               'Average Specificity', 'Average Communication',
                                               'Average Problem-Solving',
                                               'Average Professionalism', 'Average Adaptability'])

                    # Display metrics based on selected criteria
                    if selected_criteria == 'Total Score':
                        average_score = df['Average Score'].max()
                        best_candidate = df.loc[df['Average Score'].idxmax(), 'Candidate']
                        df_2 = df[['Candidate', 'Average Score']]
                        df_2.sort_values('Average Score', ascending=False, inplace=True, ignore_index=True)
                        df_2.index = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
                        df_2['Average Score'] = df_2['Average Score'].apply(remove_trailing_zeroes)
                        styled_df = apply_gradient_color(df_2, 'Average Score', 0, 17.5, 35)
                        st.dataframe(styled_df)
                        st.metric(label="Best Candidate", value=best_candidate)
                        st.metric(label="Average Score", value=f"{average_score:.2f}")
                    else:
                        criteria_names = {
                            'Total Score': 'Average Score',  # Special case for total score
                            'Relevance to Question': 'Average Relevance',
                            'Use of STAR Method': 'Average STAR Method',
                            'Specificity and Detail': 'Average Specificity',
                            'Communication Skills': 'Average Communication',
                            'Problem-Solving and Critical Thinking': 'Average Problem-Solving',
                            'Professionalism and Composure': 'Average Professionalism',
                            'Adaptability and Learning': 'Average Adaptability'
                        }

                        criteria_column = criteria_names[selected_criteria]
                        average_score = df[criteria_column].max()
                        best_candidate = df.loc[df[criteria_column].idxmax(), 'Candidate']
                        df_2 = df[['Candidate', criteria_column]]
                        df_2.sort_values(criteria_column, ascending=False, inplace=True, ignore_index=True)
                        df_2.index = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
                        df_2[criteria_column] = df_2[criteria_column].apply(remove_trailing_zeroes)
                        styled_df = apply_gradient_color(df_2, criteria_column, 0, 2.5, 5)
                        st.dataframe(styled_df)
                        st.metric(label=f"Best Candidate ({selected_criteria})", value=best_candidate)
                        st.metric(label=f"Average {selected_criteria} Score", value=f"{average_score:.2f}")


                    # Plotly horizontal bar chart
                    fig = px.bar(df.melt(id_vars='Candidate',
                                         value_vars=['Average Relevance', 'Average STAR Method', 'Average Specificity',
                                                     'Average Communication', 'Average Problem-Solving',
                                                     'Average Professionalism', 'Average Adaptability'],
                                         var_name='Criteria',
                                         value_name='Score'),
                                 x='Score',
                                 y='Criteria',
                                 color='Candidate',
                                 orientation='h',
                                 title="Candidates' Scores Across Different Criteria",
                                 labels={'Score': 'Score', 'Criteria': 'Criteria', 'Candidate': 'Candidate'},
                                 range_x=[0, 5])  # Set x-axis range from 0 to 5

                    st.plotly_chart(fig)
                else:
                    st.write("No candidates found with the specified scores.")

    # Logout Button for Interviewer
    if st.button("Logout"):
        st.session_state.role = None
        st.session_state.username = None
        st.experimental_rerun()
