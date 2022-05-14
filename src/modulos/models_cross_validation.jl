include("rna.jl");
include("bondad.jl");

using Random;
using ScikitLearn;

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


function holdOut(N::Int, P::Real)
	index = randperm(N);
	ntest = floor(Int,N*P);
	index[1:ntest], index[ntest:end];
end;

function holdOut(N::Int, Ptest::Real, Pval::Real)
	itrain, itest = holdOut(N, Ptest);
	nval = floor(Int, N*Pval);
	itrain[nval:end], itest, itrain[1:nval];
end;

function subArray(dataset::AbstractArray{<:Float32,2},
	 			indexes::Array{Int64,1})
	subArr = Array{Float32, 2}(undef, size(indexes, 1), size(dataset, 2))
	for i in 1:length(indexes)
		subArr[i, :] = dataset[indexes[i], :]
	end
	return subArr
end;

function crossvalidation(numPatrones::Int, numConj::Int)

	vConj = collect(1:numConj)
	val = ceil(Int64, numPatrones/numConj)
	vPatrones = repeat(vConj,val)
	vPatrones = getindex(vPatrones, 1:numPatrones)
	Random.shuffle!(vPatrones)
end;


function crossvalidation(targets::AbstractArray{Bool,2}, subconj::Int)

	m,n = size(targets)
	indices = zeros(Int, m)
	
	for n in eachcol(targets)
		indices[n] = crossvalidation(sum(n), subconj)
	end
	
	return indices
end;
	
	
function crossvalidation(targets::AbstractArray{<:Any, 1}, subconj::Int)

	targets2 = oneHotEncoding(targets)
	crossvalidation(targets2, subconj)
end;


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::Array{Float64,2}, targets::Array{<:Any,1}, numFolds::Int64)
    @assert(size(inputs,1)==length(targets));
    Random.seed!(10);
    
    crossValidationIndices = crossvalidation(size(inputs,1), numFolds);
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1 = Array{Float64,1}(undef, numFolds);

    for numFold in 1:numFolds
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)
            trainingInputs = inputs[crossValidationIndices.!=numFold,:];
            testInputs = inputs[crossValidationIndices.==numFold,:];
            trainingTargets = targets[crossValidationIndices.!=numFold,:];
            testTargets = targets[crossValidationIndices.==numFold,:];
        
            # Seleccionamos algoritmo
            if modelType==:SVM
                model = 
                    if haskey(modelHyperparameters, "kernel")
                        if haskey(modelHyperparameters, "kernelDegree")
                            if haskey(modelHyperparameters, "kernelGamma")
                                if haskey(modelHyperparameters, "C")
                                    SVC(
                                        kernel = modelHyperparameters["kernel"],
                                        degree = modelHyperparameters["kernelDegree"],
                                        gamma = modelHyperparameters["kernelGamma"],
                                        C = modelHyperparameters["C"]
                                    )
                                else
                                    SVC(
                                        kernel = modelHyperparameters["kernel"],
                                        degree = modelHyperparameters["kernelDegree"],
                                        gamma = modelHyperparameters["kernelGamma"],
                                    )
                                end
                            else
                                if haskey(modelHyperparameters, "C")
                                    SVC(
                                        kernel = modelHyperparameters["kernel"],
                                        degree = modelHyperparameters["kernelDegree"],
                                        C = modelHyperparameters["C"]
                                    )
                                else
                                    SVC(
                                        kernel = modelHyperparameters["kernel"],
                                        degree = modelHyperparameters["kernelDegree"],
                                    )
                                end
                            end
                        else
                            if haskey(modelHyperparameters, "kernelGamma")
                                if haskey(modelHyperparameters, "C")
                                    SVC(
                                        kernel = modelHyperparameters["kernel"],
                                        gamma = modelHyperparameters["kernelGamma"],
                                        C = modelHyperparameters["C"]
                                    )
                                else
                                    SVC(
                                        kernel = modelHyperparameters["kernel"],
                                        gamma = modelHyperparameters["kernelGamma"],
                                    )
                                end
                            else
                                if haskey(modelHyperparameters, "C")
                                    SVC(
                                        kernel = modelHyperparameters["kernel"],
                                        C = modelHyperparameters["C"]
                                    )
                                else
                                    SVC(
                                        kernel = modelHyperparameters["kernel"],
                                    )
                                end
                            end
                        end
                    else
                        if haskey(modelHyperparameters, "kernelDegree")
                            if haskey(modelHyperparameters, "kernelGamma")
                                if haskey(modelHyperparameters, "C")
                                    SVC(
                                        degree = modelHyperparameters["kernelDegree"],
                                        gamma = modelHyperparameters["kernelGamma"],
                                        C = modelHyperparameters["C"]
                                    )
                                else
                                    SVC(
                                        degree = modelHyperparameters["kernelDegree"],
                                        gamma = modelHyperparameters["kernelGamma"],
                                    )
                                end
                            else
                                if haskey(modelHyperparameters, "C")
                                    SVC(
                                        degree = modelHyperparameters["kernelDegree"],
                                        C = modelHyperparameters["C"]
                                    )
                                else
                                    SVC(
                                        degree = modelHyperparameters["kernelDegree"],
                                    )
                                end
                            end
                        else
                            if haskey(modelHyperparameters, "kernelGamma")
                                if haskey(modelHyperparameters, "C")
                                    SVC(
                                        gamma = modelHyperparameters["kernelGamma"],
                                        C = modelHyperparameters["C"]
                                    )
                                else

                                end
                            else
                                if haskey(modelHyperparameters, "C")
                                    SVC(
                                        C = modelHyperparameters["C"]
                                    )
                                else
                                    SVC()
                                end
                            end
                        end
                    end;

            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            trainingTargets = (size(trainingTargets,2) == 1) ? trainingTargets[:,1] : trainingTargets;
            model = fit!(model, trainingInputs, trainingTargets);
            testOutputs = predict(model, testInputs);
            testTargets = (size(testTargets,2) == 1) ? testTargets[:,1] : testTargets;
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);
        else
            @assert(modelType==:ANN);

            classes = unique(targets);
            if !modelHyperparameters["normalized"]
                targets = oneHotEncoding(targets, classes);
            end;

            trainingInputs = inputs[crossValidationIndices.!=numFold,:];
            testInputs = inputs[crossValidationIndices.==numFold,:];
            trainingTargets = targets[crossValidationIndices.!=numFold,:];
            testTargets = targets[crossValidationIndices.==numFold,:];
            
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            
            for numTraining in 1:modelHyperparameters["numExecutions"]
                if modelHyperparameters["validationRatio"]>0
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    ann = entrenarClassRNA(modelHyperparameters["topology"], (trainingInputs[trainingIndices,:], trainingTargets[trainingIndices,:]),
                        (testInputs, testTargets), (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                        modelHyperparameters["maxEpochs"], modelHyperparameters["minLoss"], modelHyperparameters["learningRate"], modelHyperparameters["maxEpochsVal"])[1];
                else
                    ann = entrenarClassRNA(modelHyperparameters["topology"], (trainingInputs, trainingTargets),
                        modelHyperparameters["maxEpochs"], modelHyperparameters["minLoss"], modelHyperparameters["learningRate"])[1];
                end;

                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _,
                    testF1EachRepetition[numTraining], _) =
                        if !modelHyperparameters["normalized"]
                            confusionMatrix(testTargets, ann(testInputs')');
                        else
                            confusionMatrix(collect(Int, testTargets)[:,1], ann(testInputs')[1,:], modelHyperparameters["umbral"]);
                        end;
                
            end;

            acc = mean(testAccuraciesEachRepetition);
            F1 = mean(testF1EachRepetition);
        end;

        testAccuracies[numFold] = acc;
        testF1[numFold] = F1;
        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy:
                ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");
    end;

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold
            crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ",
            100*std(testAccuracies));
    
    println(modelType, ": Average test F1 on a ", numFolds, "-fold
            crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ",
            100*std(testF1));
    
    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));
end;
