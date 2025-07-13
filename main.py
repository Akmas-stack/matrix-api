from flask import Flask, request, jsonify
from flask_cors import CORS  # добавили
import numpy as np
from scipy.linalg import svd, eig
from numpy.linalg import eigvals
from sympy import symbols, sympify, integrate
from sympy.core.sympify import SympifyError

app = Flask(__name__)
CORS(app)  # включили поддержку CORS

@app.route("/svd", methods=["POST"])
def compute_svd():
    data = request.get_json()
    matrix = np.array(data["matrix"], dtype=float)

    U, s, VT = svd(matrix, full_matrices=False)
    S = np.diag(s)



    return jsonify({
        "U": U.tolist(),
        "S": S.tolist(),
        "VT": VT.tolist()
    })
    
    
@app.route('/singular-values', methods=['POST'])
def compute_singular_values():
    data = request.get_json()
    matrix = np.array(data['matrix'])

    # Только сингулярные значения
    _, s, _ = svd(matrix, full_matrices=False)
    
    return jsonify({
        "singular_values": s.tolist()
    })
    
@app.route("/eigen", methods=["POST"])
def compute_eigen():
    data = request.get_json()
    matrix = np.array(data["matrix"], dtype=float)

    if matrix.shape[0] != matrix.shape[1]:
        return jsonify({"error": "Матрица должна быть квадратной"}), 400

    eigenvalues, eigenvectors = eig(matrix)
    norms = np.linalg.norm(eigenvectors, axis=0)
    norms[norms == 0] = 1  # избегаем деления на 0
    eigenvectors = eigenvectors / norms

    # Преобразуем комплексные числа в строки
    eigenvalues_str = [str(val) for val in eigenvalues]
    eigenvectors_str = [[str(val) for val in row] for row in eigenvectors]

  # Расчёт A * v - λ * v для каждого собственного вектора и значения (для проверки правильности вычислений)
    results = []
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        Av = matrix @ v
        lambda_v = lam * v
        diff = Av - lambda_v
        diff_str = [str(val) for val in diff]
        results.append(diff_str)

    return jsonify({
        "eigenvalues": eigenvalues_str,
        "eigenvectors": eigenvectors_str,
        "Av_minus_lambda_v": results
     })
     
     
@app.route("/spectral-radius", methods=["POST"])
def spectral_radius():
    data = request.get_json()
    matrix = np.array(data["matrix"], dtype=float)

    if matrix.shape[0] != matrix.shape[1]:
        return jsonify({"error": "Матрица должна быть квадратной"}), 400

    eigenvalues = eigvals(matrix)
    radius = max(abs(eigenvalues))

    return jsonify({
        "spectral_radius": radius,
        "eigenvalues": [str(val) for val in eigenvalues]  # Опционально: возвращаем значения
    })
    
    
    
@app.route("/qr-decomposition", methods=["POST"])
def qr_decomposition():
    data = request.get_json()
    matrix = np.array(data["matrix"], dtype=float)

    try:
        Q, R = np.linalg.qr(matrix)

        # Преобразуем результат в строки для читаемости (если нужно)
       # Q_str = [[str(val) for val in row] for row in Q]
       # R_str = [[str(val) for val in row] for row in R]

        return jsonify({
            "Q": Q.tolist(),
            "R": R.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
        
        
        
@app.route("/energy", methods=["POST"])
def energy():
    data = request.get_json()
    matrix = np.array(data["matrix"], dtype=float)

    try:
        # Энергия как сумма квадратов элементов
        elementwise_energy = np.sum(matrix ** 2)

        # Энергия как сумма сингулярных значений
        U, s, Vh = svd(matrix)
        spectral_energy = np.sum(s)

        return jsonify({
            "elementwise_energy": elementwise_energy,
            "spectral_energy": spectral_energy
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/integrate', methods=['GET'])
def integrate_expr():
    expr = request.args.get('expr', '')
    variable = request.args.get('variable', 'x')
    x = symbols(variable)
    try:
        parsed_expr = sympify(expr)
        integral = integrate(parsed_expr, x)
        return jsonify({"result": str(integral)})
    except SympifyError:
        return jsonify({"error": "Ошибка разбора выражения"}), 400
    except Exception as e:
        return jsonify({"error": f"Внутренняя ошибка: {str(e)}"}), 500
    

if __name__ == "__main__":
    app.run(debug=True)
