# import streamlit as st
# import requests

# # Title
# st.title("Credit Risk Analysis")

# st.write("Enter details")

# # Inputs
# annual_inc = st.number_input("Annual Income", min_value=1000.0, value=50000.0)
# loan_amnt = st.number_input("Loan Amount", min_value=1000.0, value=20000.0)
# dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=15.0)

# # Button
# # if st.button("Check Risk"):

# #     payload = {
# #         "annual_inc": annual_inc,
# #         "loan_amnt": loan_amnt,
# #         "dti": dti
# #     }

# #     try:
# #         response = requests.post(
# #             "http://127.0.0.1:8000/score",
# #             json=payload
# #         )

# #         result = response.json()

# #         st.subheader("Result")

# #         st.write(f"**Probability of Default:** {result['probability_of_default']:.2f}")
# #         st.write(f"**Decision:** {result['decision']}")

# #         st.subheader("🔍 Explanation")

# #         for reason in result["explanations"]:
# #             st.write(f"- {reason}")

# #     except Exception as e:
# #         st.error(f"Error: {e}")
# if st.button("Check Risk"):

#     payload = {
#         "annual_inc": annual_inc,
#         "loan_amnt": loan_amnt,
#         "dti": dti
#     }

#     try:
#         response = requests.post(
#             "http://127.0.0.1:8000/score",
#             json=payload,
#             timeout=10
#         )

#         if response.status_code != 200:
#             st.error(response.text)
#         else:
#             result = response.json()

#             st.subheader("Result")

#             st.write(f"**Probability of Default:** {result['probability_of_default']:.2f}")
#             st.write(f"**Credit Score:** {result['credit_score']:.0f}")
#             st.write(f"**Decision:** {result['decision']}")

#             st.subheader("🔍 Explanation")

#             for reason in result["explanations"]:
#                 st.write(f"- {reason}")

#     except Exception as e:
#         st.error(f"API Error: {e}")






import streamlit as st
import requests

st.title("Credit Risk Analysis (Real-Time ML)")

st.write("Enter borrower details")

# Inputs
borrower_id = st.number_input("Borrower ID", min_value=1, value=1)
annual_inc = st.number_input("Annual Income", min_value=1000.0, value=50000.0)
loan_amnt = st.number_input("Loan Amount", min_value=1000.0, value=20000.0)
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=15.0)

if st.button("Check Risk"):

    payload = {
        "borrower_id": int(borrower_id),
        "annual_inc": annual_inc,
        "loan_amnt": loan_amnt,
        "dti": dti
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/score",
            json=payload,
            timeout=10
        )

        # 🔥 Debug (keep this until stable)
        if response.status_code != 200:
            st.error(f"Error {response.status_code}: {response.text}")
        else:
            result = response.json()

            st.subheader("Result")

            st.write(f"**Borrower ID:** {result['borrower_id']}")
            st.write(f"**Probability of Default:** {result['probability_of_default']:.2f}")
            st.write(f"**Credit Score:** {result['credit_score']:.0f}")
            st.write(f"**Decision:** {result['decision']}")

            st.subheader("🔍 SHAP Explanation")

            for reason in result["explanations"]:
                st.write(f"- {reason}")

    except Exception as e:
        st.error(f"API Error: {e}")