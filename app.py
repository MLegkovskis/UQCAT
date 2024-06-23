from flask import Flask, request, render_template_string
import openai

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = ''

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    if request.method == 'POST':
        user_code = request.form['code']
        try:
            # Send the code to ChatGPT for analysis
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Execute the following Python code and return the plot it generates:\n\n```python\n{user_code}\n```"}
                ]
            )
            output = response['choices'][0]['message']['content'].strip()
            # Check for plot URL in the response
            if 'attachments' in response['choices'][0]['message']:
                plot_url = response['choices'][0]['message']['attachments'][0]['url']
        except Exception as e:
            output = str(e)
        return render_template_string(template, code=user_code, output=output, plot_url=plot_url)
    return render_template_string(template, code='', output='', plot_url=plot_url)

template = '''
<!doctype html>
<html>
<head>
    <title>Python Code Runner</title>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .container {
            display: flex;
            justify-content: space-around;
            align-items: flex-start;
            margin: 20px;
        }
        .box {
            border: 1px solid #ccc;
            padding: 20px;
            width: 45%;
            box-shadow: 2px 2px 10px #aaa;
            background-color: #f9f9f9;
        }
        .box h2 {
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 300px;
        }
        pre {
            white-space: pre-wrap; 
            word-wrap: break-word;
        }
        input[type="submit"] {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            background-color: #007BFF;
            color: white;
            border: none;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
        .plot {
            text-align: center;
        }
    </style>
</head>
<body>
    <h1 style="text-align:center;">Python Code Runner</h1>
    <div class="container">
        <div class="box">
            <h2>Input Code</h2>
            <form method="post">
                <textarea name="code">{{ code }}</textarea><br>
                <input type="submit" value="Run">
            </form>
        </div>
        <div class="box">
            <h2>Output</h2>
            <pre>{{ output }}</pre>
            {% if plot_url %}
                <div class="plot">
                    <h2>Generated Plot</h2>
                    <img src="{{ plot_url }}" alt="Generated Plot">
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
