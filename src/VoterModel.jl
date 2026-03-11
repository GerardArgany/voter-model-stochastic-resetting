module VoterModel
    export simulate

    function simulate(num_agents::Int, num_steps::Int, reset_sampler::Function)
        # Initialize agent states (0 and 1)
        states = rand(Bool, num_agents)

        # Simulation loop
        for step in 1:num_steps
            # Randomly select two agents
            agent1, agent2 = rand(1:num_agents, 2)

            # Update the state of agent1 to that of agent2
            states[agent1] = states[agent2]

            # Check for resetting
            if reset_sampler()  # Call the reset_sampler function
                states .= rand(Bool, num_agents)  # Reset all agents to random states
            end
        end
        return states
    end
end