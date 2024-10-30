# Producer-Consumer pattern
function producer(c::Channel)
    for i in 1:5
        put!(c, i^2)
        sleep(0.1)
    end
end

function consumer(c::Channel)
    while isopen(c)
        try
            data = take!(c)
            println("Consumed: $data")
        catch e
            if isa(e, InvalidStateException) && !isopen(c)
                break
            end
            rethrow(e)
        end
    end
end

# Usage
c = Channel(32)
@async producer(c)
consumer(c)