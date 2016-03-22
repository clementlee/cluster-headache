require "cutorch"
require "cunn"


model = nn.Sequential()

l1 = nn.Linear(1,20):cuda()
l2 = nn.Linear(20, 20):cuda()
l3 = nn.Linear(20, 1):cuda()

model:add(l1)
model:add(nn.Tanh():cuda())
model:add(l2)
model:add(nn.Tanh():cuda())
model:add(l3)

criterion = nn.MSECriterion():cuda()

for i = 1,100000 do
    local x = torch.CudaTensor(1)
    x[1] = torch.uniform(0, 6.28)
    local correct = torch.sin(x):cuda()

    local output = model:forward(x)
    criterion:forward(output, correct)
    model:zeroGradParameters()
    local critback = criterion:backward(model.output:cuda(), correct)
    model:backward(x, critback)
    
    model:updateParameters(0.01)
end


