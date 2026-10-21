package com.drdo.anc;
import jakarta.validation.Valid;
import java.util.List;
import java.util.Set;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

/** POST and GET /api/runs. */
@RestController
@RequestMapping("/api/runs")
public class RunController {
    static final Set<String> METHODS =
            Set.of("noisy", "spectral", "wiener", "lms", "nlms", "ai", "hybrid");
    private final RunRepository repo;

    public RunController(RunRepository repo) {
        this.repo = repo;
    }

    /** Save a run; 400 if method is not one of the contract values. */
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public Run create(@Valid @RequestBody Run run) {
        if (!METHODS.contains(run.method())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "invalid method");
        }
        repo.insert(run);
        return run;
    }

    /** List all runs. */
    @GetMapping
    public List<Run> list() {
        return repo.findAll();
    }
}
