using Statistics;

function accuracy(target::AbstractArray{Bool,1},
    outputs::AbstractArray{Bool,1})
    @assert size(target) == size(outputs)
    classComparison = target .== outputs
    accuracy = mean(classComparison)
end

function accuracy(target::AbstractArray{Bool,2},
    outputs::AbstractArray{Bool,2})
    @assert size(target) == size(outputs)
    if size(outputs, 2) == 1
        accuracy(reshape(target, size(target, 1)), reshape(outputs, size(outputs, 1)))
    elseif size(outputs, 2) > 2
        classComparison = target .== outputs
        correctClassifications = all(classComparison, dims=2)
        accuracy = mean(correctClassifications)
    end
end

function accuracy(target::AbstractArray{Bool,1},
    outputs::AbstractArray{<:Real,1}, threshold = 0.5)
    @assert size(target) == size(outputs)
    out = outputs .>= threshold
    accuracy(target, out)
end

function accuracy(target::AbstractArray{Bool,2},
    outputs::AbstractArray{<:Real,2}, threshold = 0.5)
    @assert size(target) == size(outputs)
    if size(outputs, 2) == 1
        accuracy(reshape(target, size(target, 1)), reshape(outputs, size(outputs, 1)))
    elseif size(outputs, 2) > 2
        classifiedOut= classifyOutputs(outputs, threshold)
        accuracy(target, classifiedOut)
    end
end

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

	vaux = outputs .== targets;
	vp = 0; vn = 0; fp = 0; fn = 0;
	
	verd = findall(vaux)
	pos = findall(targets)
	
	vp = length(findall(vaux .& targets));
	vn = length(verd) - vp;
	fp = length(pos) - vp;
	fn = length(vaux) - (vp+vn+fp);
	
	accuracy0 = if vn+vp+fn+fp != 0 (vn+vp)/(vn+vp+fn+fp) else 0 end;
	error_rate0 = if vn+vp+fn+fp != 0 (fn+fp)/(vn+vp+fn+fp) else 0 end;
	sensitivity0 = if fn+vp != 0 vp/(fn+vp) else 0 end;
	specificity0 = if vn+fp != 0 vn/(vn+fp) else 0 end;
	pos_pred_val0 = if vp+fp != 0 vp/(vp+fp) else 0 end;
	neg_pred_val0 = if vn+fn != 0 vn/(vn+fn) else 0 end;
	F1score0 = if sensitivity0 + pos_pred_val0 != 0
		2*(sensitivity0 * pos_pred_val0) / (sensitivity0 + pos_pred_val0) else 0 end;
	confM = [vn fp; fn vp];
	
	return (accuracy0, error_rate0, sensitivity0, specificity0, pos_pred_val0, neg_pred_val0, F1score0, confM)
end;
			
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{<:Real,1}, umbral::Real)

	vaux1 = collect(Bool, outputs .> umbral);
	vaux2 = collect(Bool, targets .> umbral);

	confusionMatrix(vaux1, vaux2);
	
end;

function confusionMatrix(outputs::AbstractArray{Bool,2},
	targets::AbstractArray{Bool,2}, weighted::Bool)
	
	@assert(all([in(output, unique(targets)) for output in outputs]));

	numClasses = size(targets, 2);
	numInstances =size(targets, 1);

	if numClasses == 1
		return confusionMatrix(outputs[:,1], target[:,1]);
	end;

	matrix = zeros(numClasses, numClasses);

	sensibilidades = Array{Float32, 1}(undef, numClasses);
	especificidades = Array{Float32, 1}(undef, numClasses);
	VPPs = Array{Float32, 1}(undef, numClasses);
	VPNs  = Array{Float32, 1}(undef, numClasses);
	F1s = Array{Float32, 1}(undef, numClasses);

	for i in 1:numClasses
		_,_,sensibilidades[i],especificidades[i],VPPs[i],VPNs[i],F1s[i],_ =
			confusionMatrix(outputs[:,i], targets[:,i]);
	end;

	for i in 1:numInstances
		y = findfirst(outputs[i,:]);
		x = findfirst(targets[i,:]);

		matrix[y,x] += 1;
	end

	precision = accuracy(targets, outputs);

	if weighted
		ponderacion = mapslices(r -> count(r)/numInstances, targets, dims=1);

		return precision,
			1-precision,
			mean(sensibilidades.*ponderacion),
			mean(especificidades.*ponderacion),
			mean(VPPs.*ponderacion),
			mean(VPNs.*ponderacion),
			mean(F1s.*ponderacion),
			matrix;
	else
		return precision,
			1-precision,
			mean(sensibilidades),
			mean(especificidades),
			mean(VPPs),
			mean(VPNs),
			mean(F1s),
			matrix;
	end;
end;
