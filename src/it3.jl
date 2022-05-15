include("modulos/bondad.jl");
include("modulos/datasetImages.jl");
include("modulos/datasets.jl");
include("modulos/models_cross_validation.jl");
include("modulos/rna.jl");

using BSON: @load
using Plots;

# Codificación BD
eyeData, notEyeData =
	eyeImagesToDatasets("BBDD/papa_noel/eye", "BBDD/papa_noel/not-a-eye");
inDS, outDS = randDataset(eyeData, notEyeData);

# Repartición del dataset
nInstXset = convert(Int, floor(size(inDS, 1)/3));
trainset = inDS[1:nInstXset,:], outDS[1:nInstXset];
testset = inDS[(nInstXset+1):(nInstXset*2),:], outDS[(nInstXset+1):(nInstXset*2)];
validset = inDS[(nInstXset*2+1):end,:], outDS[(nInstXset*2+1):end];

# Entrenamiento
eye_trained_chain, lossestrain, lossestest, lossesvalid =
	entrenarClassRNA([16,8], trainset, testset, validset, 100, 0, 0.01);

# Errores
plosses = plot(lossestrain, label="Entrenamiento")
plot!(plosses,lossestest, label="Test")
plot!(plosses,lossesvalid, label="Validación")

# Probar un caso
test = eyeImagesToDatasets(loadImage("BBDD/papa_noel/eye/10.Santa.jpg")[2])
prueba = eye_trained_chain(eyeImagesToDatasets(loadImage("BBDD/papa_noel/eye/10.Santa.jpg")[2]))

# Curva ROC para la R.N.A. de detección de ojos
curvaROC = map(umbral ->(
		m = confusionMatrix(convert(Array{Int}, outDS), eye_trained_chain(inDS')[1,:], umbral)[8];
		(m[1,2]/sum(m[1,1:2]),m[2,2]/sum(m[2,1:2]))
	), 0:0.01:1);
pROC = plot(curvaROC)

# Matriz de confusión para la R.N.A. de detección de ojos
acc, e_rate, sens, spec, ppv, npv, F1, confM =
	confusionMatrix(convert(Array{Int}, outDS), eye_trained_chain(inDS')[1,:], 0.5);


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


# Se carga la R.N.A. (Se requiere haber ejecutado it2.jl previamente)
@load "mymodel.bson" trained_chain

imageToDataX = if size(params(trained_chain)[1],2) == 5
		imageToData3
	elseif size(params(trained_chain)[1],2) == 7
		imageToData2
	else
		imageToData
	end;

# Curva ROC para la combinación del sistema de detección de ojos y Santa
folders = "BBDD/papa_noel/human", "BBDD/papa_noel/not-a-human";
pROC = plot()

for uErrors in 5:5
	println(uErrors);
	for uRNA in 0.01:0.098:0.99
		println(uRNA);

		curvaROC = convert(Array{Tuple{Float64,Float64},1}, []);
		for uDist in 0:0.1:1
			outputs = [];
			targets = [];
			
			println(uDist);
			for i in 1:2
				for fileName in readdir(folders[i])
					if isImageExtension(fileName)
						push!(outputs, testRNAfaceImage(string(folders[i], "/", fileName), x -> eye_trained_chain(x), uRNA, uDist, 5) &
							(trained_chain(imageToDataX(string(folders[i], "/", fileName)))[1] > 0.5));
						push!(targets, i==1);
					end;
				end;
			end;

			m = confusionMatrix(convert(Array{Bool,1}, outputs), convert(Array{Bool,1}, targets))[8];

			println(m)

			push!(curvaROC, (m[1,2]/sum(m[1,1:2]),m[2,2]/sum(m[2,1:2])));
		end;
		plot!(pROC, curvaROC, label=string("uRNA: ",uRNA,", uErrors: ",uErrors))
	end
end

pROC

# Matriz de confusión para la combinación del sistema de detección de ojos y Santa
# con la base de datos de fotos de humanos y no humanos
folders = "BBDD/papa_noel/human", "BBDD/papa_noel/not-a-human";
outputs = [];
targets = [];

for i in 1:2
   for fileName in readdir(folders[i])
       if isImageExtension(fileName)
               push!(outputs, testRNAfaceImage(string(folders[i], "/", fileName), x -> eye_trained_chain(x), 0.8, 0.3, 5) &
               (trained_chain(imageToDataX(string(folders[i], "/", fileName)))[1] > 0.5));
               push!(targets, i==1);
       end;
   end;
end;

confusionMatrix(convert(Array{Bool,1}, outputs), convert(Array{Bool,1}, targets))

# Matriz de confusión para solo el sistema de detección de Santa
# con la base de datos de fotos de humanos y no humanos
folders = "BBDD/papa_noel/human", "BBDD/papa_noel/not-a-human";
outputs = [];
targets = [];

for i in 1:2
	for fileName in readdir(folders[i])
		if isImageExtension(fileName)
			push!(outputs, trained_chain(imageToDataX(string(folders[i], "/", fileName)))[1] > 0.5);
			push!(targets, i==1);
		end;
	end;
end;

confusionMatrix(convert(Array{Bool,1}, outputs), convert(Array{Bool,1}, targets))
