include("modulos/bondad.jl");
include("modulos/datasetImages.jl");
include("modulos/datasets.jl");
include("modulos/models_cross_validation.jl");
include("modulos/rna.jl");

using BSON: @save
using Plots;

# Codificación BD
santaData, notSantaData =
	santaImagesToDatasets3("BBDD/papa_noel/santa", "BBDD/papa_noel/not-a-santa");
inDS, outDS = randDataset(santaData, notSantaData);

# Repartición del dataset
nInstXset = convert(Int, floor(size(inDS, 1)/3));
trainset = inDS[1:nInstXset,:], outDS[1:nInstXset];
testset = inDS[(nInstXset+1):(nInstXset*2),:], outDS[(nInstXset+1):(nInstXset*2)];
validset = inDS[(nInstXset*2+1):end,:], outDS[(nInstXset*2+1):end];

# Entrenamiento
trained_chain, lossestrain, lossestest, lossesvalid =
	entrenarClassRNA([16,8], trainset, testset, validset, 100, 0, 0.01);

# Errores
plosses = plot(lossestrain, label="Entrenamiento")
plot!(plosses,lossestest, label="Test")
plot!(plosses,lossesvalid, label="Validación")

# Probar un caso
test = imageToData3("BBDD/papa_noel/santa/1.Santa.jpg");
prueba = trained_chain(test);

# Curva ROC
curvaROC = map(umbral ->(
		m = confusionMatrix(convert(Array{Int}, outDS), trained_chain(inDS')[1,:], umbral)[8];
		(m[1,2]/sum(m[1,1:2]),m[2,2]/sum(m[2,1:2]))
	), 0:0.01:1);
pROC = plot(curvaROC)

# Matriz de confusión
acc, e_rate, sens, spec, ppv, npv, F1, confM =
	confusionMatrix(convert(Array{Int}, outDS), trained_chain(inDS')[1,:], 0.5);

# Guardar R.N.A.
@save "mymodel.bson" trained_chain


# Models cross validation

# R.N.A.
parameters = Dict("numExecutions"=>10, "validationRatio"=>0.25,
	"topology"=>[32,16], "maxEpochs"=>200, "learningRate"=>0.01,
	"minLoss"=>0, "maxEpochsVal"=>10, "normalized"=>true, "umbral"=>0.5);
modelCrossValidation(:ANN, parameters, inDS, outDS, 10)

parameters = Dict("numExecutions"=>10, "validationRatio"=>0.25,
	"topology"=>[16,8], "maxEpochs"=>100, "learningRate"=>0.01,
	"minLoss"=>0, "maxEpochsVal"=>10, "normalized"=>true, "umbral"=>0.5);
modelCrossValidation(:ANN, parameters, inDS, outDS, 10)

parameters = Dict("numExecutions"=>10, "validationRatio"=>0.25,
	"topology"=>[8,4], "maxEpochs"=>100, "learningRate"=>0.02,
	"minLoss"=>0, "maxEpochsVal"=>10, "normalized"=>true, "umbral"=>0.5);
modelCrossValidation(:ANN, parameters, inDS, outDS, 10)

parameters = Dict("numExecutions"=>10, "validationRatio"=>0.25,
	"topology"=>[4,2], "maxEpochs"=>100, "learningRate"=>0.01,
	"minLoss"=>0, "maxEpochsVal"=>10, "normalized"=>true, "umbral"=>0.5);
modelCrossValidation(:ANN, parameters, inDS, outDS, 10)

parameters = Dict("numExecutions"=>10, "validationRatio"=>0.25,
	"topology"=>[32], "maxEpochs"=>100, "learningRate"=>0.01,
	"minLoss"=>0, "maxEpochsVal"=>10, "normalized"=>true, "umbral"=>0.5);
modelCrossValidation(:ANN, parameters, inDS, outDS, 10)

parameters = Dict("numExecutions"=>10, "validationRatio"=>0.25,
	"topology"=>[16], "maxEpochs"=>100, "learningRate"=>0.01,
	"minLoss"=>0, "maxEpochsVal"=>10, "normalized"=>true, "umbral"=>0.5);
modelCrossValidation(:ANN, parameters, inDS, outDS, 10)

parameters = Dict("numExecutions"=>10, "validationRatio"=>0.25,
	"topology"=>[8], "maxEpochs"=>100, "learningRate"=>0.01,
	"minLoss"=>0, "maxEpochsVal"=>10, "normalized"=>true, "umbral"=>0.5);
modelCrossValidation(:ANN, parameters, inDS, outDS, 10)

parameters = Dict("numExecutions"=>10, "validationRatio"=>0.25,
	"topology"=>[4], "maxEpochs"=>100, "learningRate"=>0.01,
	"minLoss"=>0, "maxEpochsVal"=>10, "normalized"=>true, "umbral"=>0.5);
modelCrossValidation(:ANN, parameters, inDS, outDS, 10)

# Arboles de decision
parameters = Dict();
parameters["maxDepth"] = 2;
modelCrossValidation(:DecisionTree, parameters, inDS, outDS, 10)

parameters = Dict();
parameters["maxDepth"] = 4;
modelCrossValidation(:DecisionTree, parameters, inDS, outDS, 10)

parameters = Dict();
parameters["maxDepth"] = 8;
modelCrossValidation(:DecisionTree, parameters, inDS, outDS, 10)

parameters = Dict();
parameters["maxDepth"] = 16;
modelCrossValidation(:DecisionTree, parameters, inDS, outDS, 10)

parameters = Dict();
parameters["maxDepth"] = 32;
modelCrossValidation(:DecisionTree, parameters, inDS, outDS, 10)

parameters = Dict();
parameters["maxDepth"] = 64;
modelCrossValidation(:DecisionTree, parameters, inDS, outDS, 10)

# kNN
parameters = Dict();
parameters["numNeighbors"] = 2;
modelCrossValidation(:kNN, parameters, inDS, outDS, 10)

parameters = Dict();
parameters["numNeighbors"] = 4;
modelCrossValidation(:kNN, parameters, inDS, outDS, 10)

parameters = Dict();
parameters["numNeighbors"] = 8;
modelCrossValidation(:kNN, parameters, inDS, outDS, 10)

parameters = Dict();
parameters["numNeighbors"] = 16;
modelCrossValidation(:kNN, parameters, inDS, outDS, 10)

parameters = Dict();
parameters["numNeighbors"] = 32;
modelCrossValidation(:kNN, parameters, inDS, outDS, 10)

parameters = Dict();
parameters["numNeighbors"] = 64;
modelCrossValidation(:kNN, parameters, inDS, outDS, 10)

# SVM
parameters = Dict("kernel" => "rbf", "kernelGamma" => 2, "C" => 1);
modelCrossValidation(:SVM, parameters, inDS, outDS, 10)

parameters = Dict("kernel" => "rbf", "kernelGamma" => "scale", "C" => 1);
modelCrossValidation(:SVM, parameters, inDS, outDS, 10)

parameters = Dict("kernel" => "rbf", "kernelGamma" => "auto", "C" => 1);
modelCrossValidation(:SVM, parameters, inDS, outDS, 10)

parameters = Dict("kernel" => "linear", "C" => 1);
modelCrossValidation(:SVM, parameters, inDS, outDS, 10)

parameters = Dict("kernel" => "poly", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1);
modelCrossValidation(:SVM, parameters, inDS, outDS, 10)

parameters = Dict("kernel" => "poly", "kernelDegree" => 3, "kernelGamma" => "scale", "C" => 1);
modelCrossValidation(:SVM, parameters, inDS, outDS, 10)

parameters = Dict("kernel" => "poly", "kernelDegree" => 3, "kernelGamma" => "auto", "C" => 1);
modelCrossValidation(:SVM, parameters, inDS, outDS, 10)

parameters = Dict("kernel" => "sigmoid", "kernelGamma" => 2, "C" => 1);
modelCrossValidation(:SVM, parameters, inDS, outDS, 10)
