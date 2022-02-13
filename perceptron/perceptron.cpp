#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include <ctime>
#include <thread>

using namespace std;

double sigm(double x) {
	return 1 / (1 + exp(-x));
}

class Perceprtron
{
public:

	int amountLevel = 4;
	
	double learningRate;
	double momentum;
	
	double ans;
	double correctAns;
	double errorAns;
	
	vector<int> amountNeuronPerLevel;
	vector<vector<vector<double>>> weight = vector<vector<vector<double>>>(amountLevel - 1);
	vector<vector<vector<double>>> deltaWeight = vector<vector<vector<double>>>(amountLevel - 1);
	vector<vector<double>> deltaNeur;
	vector<vector<double>> neuron;

	void initialization() {
		for (int i = 0; i < amountLevel - 1; i++) {
			weight[i].assign(amountNeuronPerLevel[i], vector<double>(amountNeuronPerLevel[i + 1]));
			deltaWeight[i].assign(amountNeuronPerLevel[i], vector<double>(amountNeuronPerLevel[i + 1], 0));
		}
		for (int i = 0; i < amountLevel; i++) {
			vector<double> vec(amountNeuronPerLevel[i]);
			neuron.push_back(vec);
			deltaNeur.push_back(vec);
		}
	}

	void fillWeightRand() {
		for (int i = 0; i < amountLevel-1; i++) {
			for (int j = 0; j < amountNeuronPerLevel[i]; j++) {
				for (int k = 0; k < amountNeuronPerLevel[i + 1]; k++) {
					weight[i][j][k] = (double)((double)(rand() % 10001)/(double)10000);
				}
			}
		}
	}

	void iteration() {
		for (int level = 1; level < amountLevel; level++) {
			for (int curNeuron = 0; curNeuron < amountNeuronPerLevel[level]; curNeuron++) {
				for (int curWeight = 0; curWeight < amountNeuronPerLevel[level - 1]; curWeight++) {
					neuron[level][curNeuron] += neuron[level - 1][curWeight] * weight[level - 1][curWeight][curNeuron];
				}
				neuron[level][curNeuron] = sigm(neuron[level][curNeuron]);
			}
		}
		ans = neuron[amountLevel - 1][0];
		errorAns = (double)((correctAns - ans) * (correctAns - ans));
	}

	void inputdata(vector<double> &dataSet) {
		for (int curNeuron = 0; curNeuron < amountNeuronPerLevel[0]; curNeuron++) 
			neuron[0][curNeuron] = dataSet[curNeuron];
		correctAns = dataSet[dataSet.size() - 1];
	}

	void learning() {
		for (int curNeur = 0; curNeur < amountNeuronPerLevel[amountLevel - 1]; curNeur++) {
			deltaNeur[amountLevel - 1][curNeur] = (correctAns - ans) * ((1 - ans) * ans);
		}

		for (int level = amountLevel - 2; level >= 1; level--) {
			for (int curNeur = 0; curNeur < amountNeuronPerLevel[level]; curNeur++) {

				double sumWeightOnDelta = 0;
				for (int targetNeur = 0; targetNeur < amountNeuronPerLevel[level + 1]; targetNeur++)
					sumWeightOnDelta += deltaNeur[level + 1][targetNeur] * weight[level][curNeur][targetNeur];

				deltaNeur[level][curNeur] = ((1 - neuron[level][curNeur]) * neuron[level][curNeur]) * sumWeightOnDelta;
			}
		}

		for (int level = amountLevel - 2; level >= 0; level--) {
			for (int curNeur = 0; curNeur < amountNeuronPerLevel[level]; curNeur++) {
				for (int curWeight = 0; curWeight < amountNeuronPerLevel[level + 1]; curWeight++) {
					double gradient = neuron[level][curNeur] * deltaNeur[level + 1][curWeight];

					deltaWeight[level][curNeur][curWeight] = learningRate * gradient + momentum * deltaWeight[level][curNeur][curWeight];
					weight[level][curNeur][curWeight] += deltaWeight[level][curNeur][curWeight];
				}
			}
		}
	}
	Perceprtron(double rate, double moment) {
		learningRate = rate;
		momentum = moment;

		ans = 0;
		correctAns = 0;
		errorAns = 0;
		amountLevel = 4;
		amountNeuronPerLevel = { 2, 3, 3, 1 };
	}
};


int main()
{
	unsigned int startTime = clock();
	cout << thread::hardware_concurrency() << endl;

	vector<vector<double>> dataSet = { {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0} };

	Perceprtron perceptron(0.5, 0.3);
	perceptron.initialization();
	perceptron.fillWeightRand();
	for (int epoch = 0; epoch <= 20000; epoch++) {
		for (int iter = 0; iter < dataSet.size(); iter++) {
			perceptron.inputdata(dataSet[iter]);
			perceptron.iteration();
			if (epoch % 20000 == 0) {
				cout << epoch<<" "<<dataSet[iter][2] << " ";
				cout << perceptron.ans << " ";
				cout << perceptron.errorAns<< " curent time:" <<clock() - startTime<< "ms" << endl;
			}	
			perceptron.learning();
		}
	}
	cout<<clock() - startTime << "ms" << endl;
}
