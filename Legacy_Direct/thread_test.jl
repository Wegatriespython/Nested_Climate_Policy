using Base.Threads

function test_thread_utilization()
    println("Number of threads available: ", Threads.nthreads())
    
    # Create a large workload to make threading visible
    n_batches = 20
    batch_size = 50
    
    for batch in 1:n_batches
        println("\nStarting batch $batch...")
        
        @threads for i in 1:batch_size
            # Simulate computational work
            thread_id = Threads.threadid()
            sleep(0.1)  # Make the work visible in task manager
            println("Thread $thread_id processing item $i in batch $batch")
        end
    end
end

# Run the test
test_thread_utilization()
