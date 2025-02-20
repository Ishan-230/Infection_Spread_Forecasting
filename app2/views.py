from django.shortcuts import render, redirect, HttpResponse
from .models import User_data  # Import the model

from django.shortcuts import render, get_object_or_404, redirect


# View for the homepage
def index(request):
    context = "yek key msg se show kr rha hai "
    return render(request, "index.html", {"key":context})


# ------------------------------------------------------------------------------------------------------------------------

# View for the signup page
def signup(request):
    if request.method == "GET":
        return render(request, "signup.html")
    
    else:
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        
        # Check if email already exists in the database
        if User_data.objects.filter(email=email).exists():
            msg = "Email already exists"
            return render(request, "signup.html", {"key": msg})
        else:
            # Create a new user and save to the database
            ob = User_data(username=username, email=email, password=password)
            ob.save()
            return redirect("/login")

# ------------------------------------------------------------------------------------------------------------------------
# View for the login page
def login(request):
    msg = ""  # Initialize msg variable
    if request.method == "POST":
        email = request.POST["email"]  
        password = request.POST["password"] 
        print(email, password)
        
        if User_data.objects.filter(email=email).exists():
            if User_data.objects.filter(email=email, password=password).exists():
                request.session["email"] = email
                return redirect("/profile")
            else:
                msg = "Password is incorrect"
                return render(request, "login.html", {"key": msg})
        else:
            msg = "Email is invalid or incorrect"
            return render(request, "login.html", {"key": msg})
    
    return render(request, "login.html", {"key": msg})  # Corrected template

# ------------------------------------------------------------------------------------------------------------------------
# View for the profile page
def profile(request):
    if "email" in request.session:
        email = request.session.get("email")
        print(email)

        data = User_data.objects.filter(email=email)
        print(data)

        return render(request, "profile.html", {"data": data})

    else:
        msg = "Please log in again."
        return render(request, "login.html", {"key": msg})



# ------------------------------------------------------------------------------------------------------------------------
# View for the logout page
def logout(request):
    if "email" in request.session:
        del request.session["email"]
        return redirect("/login")
    else:
        return redirect("/login")

# ------------------------------------------------------------------------------------------------------------------------
# View for the update page
def update(request):
    if request.method == "POST":
        # id=request.POST.get("ID")
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        User_data.objects.filter(email=email).update(username=username, password=password)
        return redirect("/login")
    else:
        if "email" in request.session:
            email = request.session.get("email")
            data = User_data.objects.filter(email=email)
            return render(request, "update.html", {"data": data})

        else:
            return HttpResponse("Please log in again")

# ------------------------------------------------------------------------------------------------------------------------
def delete(request):
    if "email" in request.session:
        email = request.session.get("email")
        ob=User_data.objects.filter(email=email)
        ob.delete()
        return redirect("/login")
    else:
        return HttpResponse("Please login again")
    
    # ------------------------------------------------------------------------------------------------------------------------


def datainsight(request):
    return render(request, 'datainsight.html')

def aboutus(request):
    return render(request, 'aboutus.html')

def contactus(request):
    return render(request, 'contactus.html')

# ------------------------------------------------------------------------------------------------------------------------
# def forecasting(request):
#     # Load your disease data
#     data = {
#         'Disease': ['Dengue', 'Malaria', 'Typhoid'],
#         'Cases': [1200, 850, 430],
#         'Deaths': [12, 9, 4]
#     }
#     df = pd.DataFrame(data)

#     # Pass the data to the template
#     context = {
#         'disease_data': df.to_dict(orient='records')
#     }
#     return render(request, 'forecasting.html', context)


def forecasting(request):
    # Your forecasting logic here
    return render(request, 'forecasting.html')


