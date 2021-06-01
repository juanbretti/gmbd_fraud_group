# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:08:57 2020

@author: Manoel Fernando Alonso Gadi - mfalonso@faculty.ie.edu

This is a simple web service create to support the implementation of
machine learning models developed by students, in this case for
Company Credit Scoring models used of the decision of accepting or rejecting 
loan requests.

PYTHON PACKAGE REQUIREMENT - make sure you install it before hitting run:
    pip install flask_bootstrap
    pip install flask_wtf
    pip install flask_sqlalchemy
    pip install flask_login

LAST: make sure you change the working directory to be the path where the app.py is!

FINALLY: you can hit run!

"""
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, BooleanField, FloatField, SelectField

from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./database/database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'



# Support function for calculating monthly payment
import math
def calc_monthly_payment(principal, interest, months):
    '''
    given mortgage loan principal, interest(%) and years to pay
    calculate and return monthly payment amount
    '''
    # monthly rate from annual percentage rate
    interest_rate = interest/(100 * 12)
    # total number of payments
    payment_num = months
    # calculate monthly payment
    payment = principal * \
        (interest_rate/(1-math.pow((1+interest_rate), (-payment_num))))
    return payment

# Support function for dealing with numbers with comma instead of dot as decimal points separator.
class MyFloatField(FloatField):
    def process_formdata(self, valuelist):
        if valuelist:
            try:
                self.data = float(valuelist[0].replace(',', '.'))
            except ValueError:
                self.data = None
                raise ValueError(self.gettext('The value entered is not numeric'))

##############################################################################
#     SQLAlchemy ORM classes - START                                         #
##############################################################################

# SQLAlchemy ORM class for Bank
class Bank(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bank_total_assets = db.Column(db.Float,default=2000000.0)
    bank_total_borrowed = db.Column(db.Float,default=0.0)
    coef_own_capital = db.Column(db.Float,default=10.0)
    coef_ebitda = db.Column(db.Float,default=33.0)
    coef_concentration = db.Column(db.Float,default=5.0)

bank = Bank.query.filter_by(id=1).first() #Loading the first line of Bank dataset to be the bank object.


# SQLAlchemy ORM class for users
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))
    profile = db.Column(db.String(80), default='customer')

# SQLAlchemy ORM class for loans
class Loan(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    loan_amount = db.Column(db.Float())
    number_of_installments = db.Column(db.Integer())
    username = db.Column(db.String(15))
    nif = db.Column(db.String(9))
    status = db.Column(db.String(15))
    monthly_payment = db.Column(db.Float()) 
    
    
# SQLAlchemy ORM class for Companies
from datetime import datetime
class Company(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nif = db.Column(db.String(9))
    name = db.Column(db.String(50))
    p10000 = db.Column(db.Float)
    p20000 = db.Column(db.Float)
    p40100_plus_40500 = db.Column(db.Float)
    p49100_plus_40800 = db.Column(db.Float)
    p31200_plus_32300 = db.Column(db.Float)
    ebitda_income = db.Column(db.Float)
    debt_ebitda = db.Column(db.Float)
    rraa_rrpp = db.Column(db.Float) # Leveraging - External Resources / Own Resources - Recursos Ajenos / recursos propios
    log_operating_income = db.Column(db.Float)
    prob_default = db.Column(db.Float)
    username = db.Column(db.String(15))    
    data_timestamp = db.Column(db.DateTime, default=datetime.utcnow)

##############################################################################
#     SQLAlchemy ORM classes - END                                           #
##############################################################################


##############################################################################
#     FlaskForm classes - START                                              #
##############################################################################

# FlaskForm class for Register Form
class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])

# FlaskForm class for Login Form
class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

# FlaskForm class for Loan Form
class LoanForm(FlaskForm):
    rowid = StringField('Order ID / ID del Pedido')
    loan_amount = MyFloatField('Loan Amount / Valor del Prestámo ', validators=[InputRequired()])
    number_of_installments = SelectField('Number of Installments / Número de Pagos', choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'),('5', '5'), ('6', '6'), ('7', '7'), ('8', '8')])        
    nif = StringField('Id of the Company / NIF de su Empresa', validators=[InputRequired(), Length(min=9, max=9)],default='')    

# FlaskForm class for Company Form
class CompanyForm(FlaskForm):
#    rowid = StringField('ID')
    orderid = StringField('Order ID / ID del Pedido')     
    loan_amount = MyFloatField('Loan Amount / Valor del Prestámo', validators=[InputRequired()])
    number_of_installments = SelectField('Number of Installments / Número de Pago', choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'),('5', '5'), ('6', '6'), ('7', '7'), ('8', '8')])           
    nif = StringField('Id of the Company / NIF de su Empresa', validators=[InputRequired(), Length(min=9, max=9)],default='')    
    name = StringField('Name of your Company / Nombre/Razón Social de su Empresa', validators=[InputRequired(), Length(min=3, max=50)])
    p40100_plus_40500 = MyFloatField('Operating Income / Ingresos', validators=[InputRequired()])
    p49100_plus_40800 = MyFloatField('EBITDA', validators=[InputRequired()])
    p10000 = MyFloatField('Total Assets / Total activos', validators=[InputRequired()])
    p20000 = MyFloatField('Own Capital / Patrimonio neto', validators=[InputRequired()])
    p31200_plus_32300 = MyFloatField('Total Debt / Deuda total', validators=[InputRequired()])


##############################################################################
#     FlaskForm classes - END                                                #
##############################################################################



##############################################################################
#     FLASK APP ROUTE DEFITION - START                                       #
##############################################################################

# flask_login support function to load logged user object from database
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))    
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/formulario_multipaso')
def formulario_multipaso():
    return render_template('formulario_multipaso.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    message=''
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('loan'))

        message = 'Invalid username or password'

    return render_template('login.html', form=form, message=message)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    message=''
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        message='New user has been created!'
        form = LoginForm()
        return render_template('login.html', form=form, message=message)
    return render_template('signup.html', form=form, message=message)



@app.route('/loan', methods=['GET', 'POST'])
@login_required
def loan():
    form = LoanForm()
    message = ''
    if request.method == 'POST':
        if form.validate_on_submit():
            try: # UPDATING DATA FROM AN EXISTING ORDER
                order = Loan.query.filter_by(id=request.form['rowid']).first()
                order.loan_amount = form.loan_amount.data
                update_or_new = 1 # 1 for update
            except: # CREATING A NEW ORDER
                update_or_new = 0 # 0 for new
            
            if update_or_new == 1: # UPDATING DATA FROM AN EXISTING ORDER
                order.number_of_installments = form.number_of_installments.data
                order.nif = form.nif.data
                order.username = current_user.username
#                order.status = 'Pdte Revisión Datos Company'

                db.session.commit()
                message = 'Data updated for order / Datos actualizado para el pedido:' + str(request.form['rowid'])
                rowid = request.form['rowid']
            else: # CREATING A NEW ORDER
                new_loan = Loan(loan_amount=form.loan_amount.data, \
                                number_of_installments=form.number_of_installments.data, \
                                username = current_user.username, \
                                nif = form.nif.data, \
                                status = 'Company Data Pending')
                db.session.add(new_loan)
                db.session.commit()
                rowid = str(new_loan.id)
#                return(str(rowid))
             #EITHER EDITION OR NEW ORDER - GO TO THE REVIEW OF DATA DE Company  
                                 
            try:
#                return("loan post - no form - 5.1")
                company = Company.query.filter(Company.username.in_([current_user.username]),Company.nif.in_([form.nif.data])).first()
#                return("loan post - no form - 5.2: "+str(company.name))
                form = CompanyForm(orderid=str(rowid), \
                                nif = company.nif, \
                                loan_amount = form.loan_amount.data, \
                                number_of_installments = form.number_of_installments.data, \
                                name = company.name, \
                                p40100_plus_40500 = company.p40100_plus_40500, \
                                p31200_plus_32300 = company.p31200_plus_32300, \
                                p49100_plus_40800 = company.p49100_plus_40800, \
                                p10000 = company.p10000, \
                                p20000 = company.p20000
                                )               
                message = ''
            except:
                form = CompanyForm(orderid=str(rowid), \
                                nif = form.nif.data, \
                                loan_amount = form.loan_amount.data, \
                                number_of_installments = form.number_of_installments.data)                      
                message = ''                    
            order_history = Loan.query.filter_by(username=current_user.username)

            return render_template('company.html', form=form, \
                                   rows=order_history, \
                                   message=message, \
                                   name=current_user.username)
                
        else:
            try: #In this case, we try to check if the user clicked to edit data of a loan
                order = Loan.query.filter_by(id=request.form['rowid']).first()
                form = LoanForm(rowid=order.id, \
                                    loan_amount = order.loan_amount, \
                                    number_of_installments = order.number_of_installments, \
                                    nif = order.nif)
            except: # In this case the user did not entered the form data correctly
                pass
    # RETRIEVING HISTORICAL DATA FOR ORDERS
    order_history = Loan.query.filter_by(username=current_user.username)
    return render_template('loan.html', form=form, \
                           rows=order_history, \
                           message=message, \
                           name=current_user.username)


import numpy as np
import pandas as pd
from joblib import load
Rating_RandomForestClassifier_model = load('./database/Rating_RandomForestClassifier.joblib') 
@app.route('/company', methods=['GET', 'POST'])
@login_required
def company(): 
    form = CompanyForm()
    message = ''
    automated_decision = 'rejected'

    if request.method == 'POST':
        if form.validate_on_submit():
            try:
                ebitda_income = (form.p49100_plus_40800.data)/(form.p40100_plus_40500.data)
            except:
                ebitda_income = 0.0

            try:    
                debt_ebitda =(form.p31200_plus_32300.data) /(form.p49100_plus_40800.data)
            except:
                debt_ebitda = 99.99

            try:                
                rraa_rrpp = (form.p10000.data - form.p20000.data) /form.p20000.data
            except:
                rraa_rrpp = 99.99
            
            try:
                log_operating_income = np.log(form.p40100_plus_40500.data)
            except:
                log_operating_income = 0.0
                
            X = pd.DataFrame({'ebitda_income': [ebitda_income], \
                              'debt_ebitda': [debt_ebitda], \
                              'rraa_rrpp' : [rraa_rrpp], \
                              'log_operating_income' : [log_operating_income]
                              })
            prob_default = Rating_RandomForestClassifier_model.predict_proba(X)[:,1]
        
            try: # UPDATING DATA OF AN EXISTING COMPANY
                company = Company.query.filter(Company.username.in_([current_user.username]),Company.nif.in_([form.nif.data])).first()
                company.nif = form.nif.data
                update_or_new = 1 # 1 for update
            except: # CREATING A NEW COMPANY
                update_or_new = 0 # 0 for new                

            if update_or_new == 1: # UPDATING DATA FROM AN EXISTING COMPANY
                company.name = form.name.data
                company.p10000 = form.p10000.data
                company.p20000 = form.p20000.data
                company.p40100_plus_40500 = form.p40100_plus_40500.data
                company.p49100_plus_40800 = form.p49100_plus_40800.data
                company.p31200_plus_32300 = form.p31200_plus_32300.data
                company.ebitda_income = ebitda_income
                company.debt_ebitda = debt_ebitda
                company.rraa_rrpp = rraa_rrpp
                company.prob_default = prob_default
                company.log_operating_income = log_operating_income 
                company.username = current_user.username
                db.session.commit()
                message = 'Company data updated!'
            else: # CREATING A NEW COMPANY
                new_company = Company(
                        nif = form.nif.data, \
                        name = form.name.data, \
                        p10000 = form.p10000.data, \
                        p20000 = form.p20000.data, \
                        p40100_plus_40500 = form.p40100_plus_40500.data, \
                        p49100_plus_40800 = form.p49100_plus_40800.data, \
                        p31200_plus_32300 = form.p31200_plus_32300.data, \
                        ebitda_income = ebitda_income, \
                        debt_ebitda = debt_ebitda, \
                        rraa_rrpp = rraa_rrpp, \
                        prob_default = prob_default, \
                        log_operating_income = log_operating_income, \
                        username = current_user.username
                        )
                db.session.add(new_company)
                db.session.commit()        
                message = 'New Company data created!'

        #Bank
        try:
            loans = Loan.query.filter(Loan.nif.in_([company.nif]), \
                                              Loan.status.in_(['acepted']))
            used_amount = sum([loan.loan_amount for loan in loans])
        except:
            used_amount = 0
            
#
        limite_acepted = min(form.p20000.data * bank.coef_own_capital / 100.0, \
                              form.p49100_plus_40800.data * bank.coef_ebitda / 100.0, \
                              bank.bank_total_assets * bank.coef_concentration / 100.0)

        automated_decision = 'acepted'

        order = Loan.query.filter_by(id=form.orderid.data).first()

        if limite_acepted - used_amount < order.loan_amount:
            automated_decision = 'rejected'
        elif prob_default > 0.00000000000000001:
            automated_decision = 'rejected'
        
        try:
            loan = Loan.query.filter(Loan.username.in_([current_user.username]),Loan.id.in_([form.orderid.data])).first()
#                return("0.1-"+Loan.status)
        except:
            loan = Loan.query.order_by(Loan.id.desc()).first()

        loan.status = automated_decision
        db.session.commit()            

    if automated_decision == 'rejected':
        message = 'Sorry, your loan has been rejected for Risks.'
        order_history = Loan.query.filter_by(username=current_user.username)
        order.status = 'rejected'
        return render_template('company.html', form=form, \
                               rows=order_history , \
                               message=message, \
                               name=current_user.username)
    else: # request acepted
        monthly_payment = calc_monthly_payment(loan.loan_amount, 7.5, \
                                               loan.number_of_installments)
        message = 'Congratulations, your loan has been accepted and with a monthly payment of: %.2f'% \
        (monthly_payment) + " €"

        order.monthly_payment = monthly_payment
        order.status = 'acepted'
        db.session.commit()            

        order_history = Loan.query.filter_by(username=current_user.username)   
        return render_template('company.html', form=form, \
                               rows=order_history , \
                               message=message, \
                               name=current_user.username)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

##############################################################################
#     FLASK APP ROUTE DEFITION - END                                       #
##############################################################################


if __name__ == '__main__':
    app.run(debug=True)

