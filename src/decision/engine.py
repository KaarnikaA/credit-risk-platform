def decide(prob, income, loan_amnt, dti, threshold):

    # hard rules to mimic business logic
    if income < 20000:
        return "REJECT (Low Income)"

    if dti > 40:
        return "REJECT (High DTI)"

    if loan_amnt > income * 0.8:
        return "REJECT (Loan too large)"

   # model based
    if prob < threshold:
        return "APPROVE"
    elif prob < threshold + 0.1:
        return "REVIEW"
    else:
        return "REJECT"