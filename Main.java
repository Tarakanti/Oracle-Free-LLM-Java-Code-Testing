package com.tara.difftest;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.ConditionalExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.expr.UnaryExpr;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.stmt.DoStmt;
import com.github.javaparser.ast.stmt.ForEachStmt;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.SwitchEntry;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Differential testing framework for LLM-generated code.
 *
 * Usage:
 *   interactive : run with no arguments
 *   batch       : run with one argument, a file with lines "Level|Topic|Question"
 */
public class Main {

    static final HttpClient HTTP = HttpClient.newHttpClient();
    static final String[] MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"};

    // Reliability weights (test behaviour matters most)
    static final double W_AST = 0.15, W_CFG = 0.15, W_DFG = 0.10, W_TEST = 0.60;

    static final String[] HEADERS = {
            "Codes", "Model", "Compiles", "AST Sim %", "CFG Sim %", "DFG Sim %",
            "Test Pass %", "Edge Pass %", "Mutation %", "Cyclomatic", "Runtime ms", "Security", "Reliability %"
    };

        // mutation testing settings
    static final int MAX_MUTANTS = 6;          // mutants made per implementation (lower = faster)
    static final int MUTANT_TIMEOUT_SEC = 3;   // a mutant running longer than this counts as killed
    static final Map<BinaryExpr.Operator, BinaryExpr.Operator> MUTATIONS = Map.ofEntries(
            Map.entry(BinaryExpr.Operator.LESS, BinaryExpr.Operator.LESS_EQUALS),
            Map.entry(BinaryExpr.Operator.LESS_EQUALS, BinaryExpr.Operator.LESS),
            Map.entry(BinaryExpr.Operator.GREATER, BinaryExpr.Operator.GREATER_EQUALS),
            Map.entry(BinaryExpr.Operator.GREATER_EQUALS, BinaryExpr.Operator.GREATER),
            Map.entry(BinaryExpr.Operator.EQUALS, BinaryExpr.Operator.NOT_EQUALS),
            Map.entry(BinaryExpr.Operator.NOT_EQUALS, BinaryExpr.Operator.EQUALS),
            Map.entry(BinaryExpr.Operator.AND, BinaryExpr.Operator.OR),
            Map.entry(BinaryExpr.Operator.OR, BinaryExpr.Operator.AND),
            Map.entry(BinaryExpr.Operator.PLUS, BinaryExpr.Operator.MINUS),
            Map.entry(BinaryExpr.Operator.MINUS, BinaryExpr.Operator.PLUS),
            Map.entry(BinaryExpr.Operator.MULTIPLY, BinaryExpr.Operator.DIVIDE),
            Map.entry(BinaryExpr.Operator.DIVIDE, BinaryExpr.Operator.MULTIPLY),
            Map.entry(BinaryExpr.Operator.REMAINDER, BinaryExpr.Operator.DIVIDE));

    static final Set<String> RISKY_CALLS = Set.of(
            "exec", "exit", "halt", "forName", "getRuntime", "setAccessible",
            "loadLibrary", "getenv", "getProperty", "delete", "deleteIfExists");
    static final Set<String> RISKY_TYPES = Set.of(
            "ProcessBuilder", "File", "FileReader", "FileWriter",
            "FileInputStream", "FileOutputStream", "Socket", "ServerSocket", "URL");

    // limits for how much of an input / output is shown in the console
    static final int MAX_LINE_CHARS = 80;
    static final int MAX_INPUT_LINES = 5;
    static final int MAX_OUTPUT_CHARS = 40;

    // ---------- data holders ----------

    static class IOContract {
        String inputFormat = "";
        String outputFormat = "";
    }

    static class ImplResult {
        int index;
        String model;
        String code;
        Path dir;
        boolean compiled;

        Map<String, Integer> astProfile, cfgProfile, dfgProfile;
        double ast = Double.NaN, cfg = Double.NaN, dfg = Double.NaN;
        int cyclomatic = -1, security = -1;

        int normalPass, edgePass, mismatches;
        long totalNanos;
        int runs;

        double testPass, edgePassRate = Double.NaN, runtimeMs = Double.NaN, reliability;

        double mutation = Double.NaN;
        List<String> normOutputs = new ArrayList<>();
    }

    // ---------- entry point ----------

    public static void main(String[] args) throws Exception {
        if (args.length > 0) {
            runBatch(Paths.get(args[0]));
            return;
        }

        Scanner scanner = new Scanner(System.in);
        String problem;
        while (true) {
            System.out.println("Enter a programming problem to solve:");
            problem = scanner.nextLine();
            if (isValidCodingQuestion(problem)) break;
            System.out.println("Please ask a coding question.\n");
        }
        runPipeline(problem, "Interactive: Q1",
                Paths.get("proof", "interactive_" + System.currentTimeMillis() + ".docx"), true);
    }

    static void runBatch(Path file) throws Exception {
        Map<String, Integer> counter = new HashMap<>();
        for (String line : Files.readAllLines(file)) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] p = line.split("\\|", 3);
            if (p.length < 3) {
                System.out.println("Skipping bad line: " + line);
                continue;
            }
            String level = p[0].trim(), topic = p[1].trim(), question = p[2].trim();
            String key = level + "_" + topic;
            int n = counter.merge(key, 1, Integer::sum);

            System.out.println("\n==================================================");
            System.out.println(level + " | " + topic + " | Q" + n + ": " + question);
            System.out.println("==================================================");
            try {
                                String label = topic + ": Q" + n;
                runPipeline(question, label, Paths.get("proof", key + "_Q" + n + ".docx"), false);
            } catch (Exception e) {
                System.out.println("Run failed: " + e.getMessage());
            }
        }
    }

    // ---------- main pipeline for one question ----------

    static void runPipeline(String problem, String label, Path outFile, boolean verbose) throws Exception {
        // compiled classes live in a temp folder that is deleted at the end
        Path workDir = Files.createTempDirectory("difftest_");
        try {
            // 1. I/O contract
            System.out.println("\nDesigning input/output contract...");
            IOContract contract = generateIOContract(problem);
            System.out.println("INPUT:  " + contract.inputFormat);
            System.out.println("OUTPUT: " + contract.outputFormat);

            // 2. Test inputs (normal + edge, tagged separately)
            List<String> normalTests = generateTestInputs(problem, contract, 5,
                    "realistic, varied, typical inputs (not extreme edge cases)");
            List<String> edgeTests = generateTestInputs(problem, contract, 4,
                    "EDGE CASES only (for example minimum/maximum sizes, empty or single-element input, zero, "
                            + "negative numbers, duplicates, boundary values), as far as the input format allows");
            List<String> allTests = new ArrayList<>(normalTests);
            allTests.addAll(edgeTests);
            int normalCount = normalTests.size();

            printTestCases("Normal test cases:", normalTests);
            printTestCases("Edge test cases:", edgeTests);

            // 3. Prompt variations
            List<String> variations = generatePromptVariations(problem, 4);
            System.out.println("\nGenerated prompt variations:");
            variations.forEach(v -> System.out.println("- " + v));

            // 4. Generate + compile implementations (failures are recorded, not fatal)
            List<ImplResult> impls = new ArrayList<>();
            for (int i = 0; i < variations.size(); i++) {
                ImplResult r = new ImplResult();
                r.index = i + 1;
                r.model = MODELS[i % MODELS.length];
                System.out.println("\nGenerating implementation " + r.index + " from " + r.model);
                try {
                    r.code = generateProgramFromLLM(variations.get(i), contract, r.model);
                    if (verbose) System.out.println(r.code);
                    r.dir = compileProgram(r.code, workDir.resolve("impl" + r.index));
                    r.compiled = r.dir != null;
                    if (!r.compiled) System.out.println("  -> Implementation " + r.index + " failed to compile");
                } catch (Exception e) {
                    System.out.println("  -> Implementation " + r.index + " failed: " + e.getMessage());
                }
                impls.add(r);
            }

            // 5. Static analysis: AST / CFG / DFG profiles, cyclomatic complexity, security scan
            for (ImplResult r : impls) {
                if (!r.compiled) continue;
                try {
                    CompilationUnit cu = StaticJavaParser.parse(r.code);
                    r.astProfile = nodeProfile(cu);
                    r.cfgProfile = controlFlowProfile(cu);
                    r.dfgProfile = dataFlowProfile(cu);
                    r.cyclomatic = cyclomatic(cu);
                    r.security = securityIssues(cu);
                } catch (Exception e) {
                    System.out.println("  -> Could not analyse implementation " + r.index + ": " + e.getMessage());
                }
            }
            for (ImplResult r : impls) {
                if (r.astProfile == null) continue;
                r.ast = peerSimilarity(impls, r, 0);
                r.cfg = peerSimilarity(impls, r, 1);
                r.dfg = peerSimilarity(impls, r, 2);
            }

            // 6. Run every compiled implementation on every test
            System.out.println("\nTest case outputs:");
            int normalCounted = 0, edgeCounted = 0;

            for (int t = 0; t < allTests.size(); t++) {
                boolean edge = t >= normalCount;
                if (t == 0) System.out.println("\nNormal test cases:");
                if (t == normalCount) System.out.println("\nEdge test cases:");

                String input = allTests.get(t);
                String[] raw = new String[impls.size()];
                String[] norm = new String[impls.size()];

                for (int i = 0; i < impls.size(); i++) {
                    ImplResult r = impls.get(i);
                    if (!r.compiled) {
                        raw[i] = "N/A (did not compile)";
                        continue;
                    }
                    long start = System.nanoTime();
                    raw[i] = runProgram(r.dir, input);
                    r.totalNanos += System.nanoTime() - start;
                    r.runs++;
                    norm[i] = normalize(raw[i]);
                    r.normOutputs.add(norm[i]);
                }

                String majority = findMajority(norm);
                boolean anyMismatch = false;
                if (majority != null) {
                    if (edge) edgeCounted++; else normalCounted++;
                    for (int i = 0; i < impls.size(); i++) {
                        ImplResult r = impls.get(i);
                        if (!r.compiled) continue;
                        if (majority.equals(norm[i])) {
                            if (edge) r.edgePass++; else r.normalPass++;
                        } else {
                            r.mismatches++;
                            anyMismatch = true;
                        }
                    }
                }

                List<String> shown = new ArrayList<>();
                for (String o : raw) {
                    shown.add(truncate(o.replace("\r", "").replace("\n", " | "), MAX_OUTPUT_CHARS));
                }
                String flag = majority == null ? "   <-- NO MAJORITY" : (anyMismatch ? "   <-- MISMATCH" : "");
                int number = edge ? t - normalCount + 1 : t + 1;
                printNumberedInput(number, input);
                System.out.println("   Outputs: " + shown + flag);
            }
                        // 6b. Mutation testing (compiles and runs small mutated copies of each program)
            for (ImplResult r : impls) {
                if (!r.compiled) continue;
                System.out.println("Running mutation testing on implementation " + r.index + "...");
                r.mutation = mutationScore(r, allTests, workDir);
            }

            // 7. Final per-implementation metrics
            int counted = normalCounted + edgeCounted;
            for (ImplResult r : impls) {
                r.testPass = counted == 0 ? 0 : (double) (r.normalPass + r.edgePass) / counted;
                r.edgePassRate = edgeCounted == 0 ? Double.NaN : (double) r.edgePass / edgeCounted;
                r.runtimeMs = r.runs == 0 ? Double.NaN : r.totalNanos / 1e6 / r.runs;
                r.reliability = (!r.compiled || r.astProfile == null) ? 0
                        : W_AST * r.ast + W_CFG * r.cfg + W_DFG * r.dfg + W_TEST * r.testPass;
            }

            // 8. Results table
            List<String[]> rows = new ArrayList<>();
            for (ImplResult r : impls) rows.add(toRow(r));

            System.out.println("\n=== RESULTS TABLE ===");
            System.out.println(tableToString(HEADERS, rows));
            System.out.println("(Tests counted: " + normalCounted + " normal, " + edgeCounted
                    + " edge. Tests with no clear majority are skipped.)");

            double avgRel = impls.stream().mapToDouble(r -> r.reliability).average().orElse(0);
            System.out.printf("%nAverage reliability: %.2f%% - %s%n", avgRel * 100, interpretReliability(avgRel));

            localizeBugs(impls);

            // 9. Save only the question + its table, as a Word document
                        saveResults(outFile, label, HEADERS, rows, impls);
            System.out.println("\nResults saved to: " + outFile.toAbsolutePath());
        } finally {
            deleteRecursively(workDir);
        }
    }

    // ---------- LLM helpers ----------

    /** One place for all Groq calls, with retry on rate limits / server errors. */
    static String callGroq(String model, double temperature, String prompt) throws Exception {
        String apiKey = System.getenv("GROQ_API_KEY");
        if (apiKey == null) throw new RuntimeException("GROQ_API_KEY environment variable not found.");

        JsonObject requestJson = new JsonObject();
        requestJson.addProperty("model", model);
        requestJson.addProperty("temperature", temperature);
        JsonObject message = new JsonObject();
        message.addProperty("role", "user");
        message.addProperty("content", prompt);
        JsonArray messages = new JsonArray();
        messages.add(message);
        requestJson.add("messages", messages);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.groq.com/openai/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(requestJson.toString()))
                .build();

        for (int attempt = 1; attempt <= 5; attempt++) {
            HttpResponse<String> response = HTTP.send(request, HttpResponse.BodyHandlers.ofString());
            int code = response.statusCode();
            if (code == 200) {
                JsonObject json = JsonParser.parseString(response.body()).getAsJsonObject();
                JsonElement content = json.getAsJsonArray("choices").get(0).getAsJsonObject()
                        .getAsJsonObject("message").get("content");
                return (content == null || content.isJsonNull()) ? "" : content.getAsString();
            }
            if (code == 429 || code >= 500) {
                System.out.println("  (Groq busy / rate limited, retrying in " + (4 * attempt) + "s...)");
                Thread.sleep(4000L * attempt);
                continue;
            }
            throw new RuntimeException("Groq API error (" + code + "): " + response.body());
        }
        throw new RuntimeException("Groq API failed after several retries.");
    }

    static boolean isValidCodingQuestion(String input) throws Exception {
        String prompt = "Is the following text a request to solve a programming/coding problem " +
                "(something that can be solved by writing code)? Answer with ONLY the single word YES or NO, nothing else. " +
                "Text: \"" + input + "\"";
        return callGroq("openai/gpt-oss-20b", 0.0, prompt).trim().toUpperCase().contains("YES");
    }

    static IOContract generateIOContract(String problem) throws Exception {
        String prompt = "Given this programming problem: \"" + problem + "\", " +
                "define a simple, precise input/output contract for a command-line Java program that solves it. " +
                "The program should read input from standard input (keyboard) and print output to standard output (screen). " +
                "Respond in EXACTLY this format, nothing else, no extra text:\n" +
                "INPUT: <one line describing exactly what the program reads from standard input>\n" +
                "OUTPUT: <one line describing exactly what the program should print to standard output>";
        String raw = callGroq("openai/gpt-oss-120b", 0.2, prompt);

        IOContract contract = new IOContract();
        for (String line : raw.split("\n")) {
            line = line.trim();
            if (line.startsWith("INPUT:")) contract.inputFormat = line.substring("INPUT:".length()).trim();
            else if (line.startsWith("OUTPUT:")) contract.outputFormat = line.substring("OUTPUT:".length()).trim();
        }
        return contract;
    }

    static List<String> generateTestInputs(String problem, IOContract contract, int numTests, String kind) throws Exception {
        String prompt = "Given this programming problem: \"" + problem + "\", " +
                "with this input format: \"" + contract.inputFormat + "\", " +
                "generate " + numTests + " test cases that are " + kind + ". " +
                "Each test case is the COMPLETE input for one program run - if the format needs multiple lines " +
                "(e.g. a count on one line and values on the next), include ALL of those lines together as one test case. " +
                "Separate each complete test case from the next using a line containing EXACTLY: ---\n" +
                "Do not add numbering, labels, markdown, or any explanation. Output only the raw test case contents and the --- separators.";
        String raw = callGroq("openai/gpt-oss-120b", 0.5, prompt).replaceAll("(?m)^```.*$", "");

        List<String> tests = new ArrayList<>();
        for (String block : raw.split("(?m)^---\\s*$")) {
            block = block.trim();
            if (!block.isEmpty()) tests.add(block);
        }
        if (tests.size() > numTests) tests = new ArrayList<>(tests.subList(0, numTests));
        return tests;
    }

    static List<String> generatePromptVariations(String original, int n) throws Exception {
        String prompt = "Rewrite this programming problem statement in " + n +
                " different ways, keeping the exact same meaning and requirements. " +
                "Original: \"" + original + "\". " +
                "Return ONLY the " + n + " rewritten versions, one per line, no numbering, no explanations.";
        String raw = callGroq("openai/gpt-oss-120b", 0.7, prompt);

        List<String> variations = new ArrayList<>();
        for (String line : raw.split("\n")) {
            line = line.trim().replaceAll("^([-*\u2022]|\\d+[.)])\\s*", "");
            if (!line.isEmpty()) variations.add(line);
        }
        while (variations.size() < n) variations.add(original);
        return new ArrayList<>(variations.subList(0, n));
    }

    static String generateProgramFromLLM(String problemPrompt, IOContract contract, String modelId) throws Exception {
        String prompt = "Write a complete, runnable Java program that solves this problem: \"" + problemPrompt + "\". " +
                "Requirements:\n" +
                "- The public class MUST be named exactly 'Solution'.\n" +
                "- It MUST have a public static void main(String[] args) method.\n" +
                "- Input: " + contract.inputFormat + "\n" +
                "- Output: " + contract.outputFormat + "\n" +
                "- Read input using Scanner or BufferedReader from System.in.\n" +
                "- Print output using System.out.\n" +
                "Return ONLY the raw Java code, no explanations, no markdown code fences.";
            for (int attempt = 1; attempt <= 3; attempt++) {
        String code = callGroq(modelId, 0.3, prompt)
                .replaceAll("```java", "").replaceAll("```", "").replace("`", "").trim();
        if (!code.isBlank()) return code;
        System.out.println("  (Empty code on attempt " + attempt + ", retrying...)");
    }
    throw new RuntimeException("Model returned empty code after 3 attempts.");
    }

    // ---------- static analysis helpers ----------

    static Map<String, Integer> nodeProfile(CompilationUnit cu) {
        Map<String, Integer> counts = new HashMap<>();
        for (Node node : cu.findAll(Node.class)) {
            counts.merge(node.getClass().getSimpleName(), 1, Integer::sum);
        }
        return counts;
    }

    static Map<String, Integer> controlFlowProfile(CompilationUnit cu) {
        Set<String> cfgTypes = Set.of("IfStmt", "ForStmt", "ForEachStmt", "WhileStmt", "DoStmt",
                "SwitchStmt", "SwitchEntry", "ReturnStmt", "BreakStmt", "ContinueStmt");
        Map<String, Integer> counts = new HashMap<>();
        for (Node node : cu.findAll(Node.class)) {
            String type = node.getClass().getSimpleName();
            if (cfgTypes.contains(type)) counts.merge(type, 1, Integer::sum);
        }
        return counts;
    }

    static Map<String, Integer> dataFlowProfile(CompilationUnit cu) {
        Map<String, Integer> profile = new HashMap<>();
        profile.put("declarations", cu.findAll(VariableDeclarator.class).size());
        profile.put("assignments", cu.findAll(AssignExpr.class).size());
        profile.put("reads", cu.findAll(NameExpr.class).size());
        profile.put("unaryOps", cu.findAll(UnaryExpr.class).size());
        return profile;
    }

    static double compareProfiles(Map<String, Integer> a, Map<String, Integer> b) {
        Set<String> keys = new HashSet<>(a.keySet());
        keys.addAll(b.keySet());
        int match = 0, total = 0;
        for (String k : keys) {
            int ca = a.getOrDefault(k, 0), cb = b.getOrDefault(k, 0);
            match += Math.min(ca, cb);
            total += Math.max(ca, cb);
        }
        return total == 0 ? 1.0 : (double) match / total;
    }

    static Map<String, Integer> profileOf(ImplResult r, int kind) {
        return kind == 0 ? r.astProfile : (kind == 1 ? r.cfgProfile : r.dfgProfile);
    }

    /** Average similarity of one implementation against all other analysable implementations. */
    static double peerSimilarity(List<ImplResult> impls, ImplResult me, int kind) {
        double total = 0;
        int count = 0;
        for (ImplResult other : impls) {
            if (other == me || other.astProfile == null) continue;
            total += compareProfiles(profileOf(me, kind), profileOf(other, kind));
            count++;
        }
        return count == 0 ? 1.0 : total / count;
    }

    /** McCabe cyclomatic complexity = 1 + number of decision points. */
    static int cyclomatic(CompilationUnit cu) {
        int c = 1;
        c += cu.findAll(IfStmt.class).size();
        c += cu.findAll(ForStmt.class).size();
        c += cu.findAll(ForEachStmt.class).size();
        c += cu.findAll(WhileStmt.class).size();
        c += cu.findAll(DoStmt.class).size();
        c += cu.findAll(CatchClause.class).size();
        c += cu.findAll(ConditionalExpr.class).size();
        for (SwitchEntry e : cu.findAll(SwitchEntry.class)) {
            if (!e.getLabels().isEmpty()) c++;
        }
        for (BinaryExpr b : cu.findAll(BinaryExpr.class)) {
            if (b.getOperator() == BinaryExpr.Operator.AND || b.getOperator() == BinaryExpr.Operator.OR) c++;
        }
        return c;
    }

    /** Counts risky API usages (process execution, file/network access, reflection, System.exit...). */
    static int securityIssues(CompilationUnit cu) {
        int n = 0;
        for (MethodCallExpr m : cu.findAll(MethodCallExpr.class)) {
            if (RISKY_CALLS.contains(m.getNameAsString())) n++;
        }
        for (ObjectCreationExpr o : cu.findAll(ObjectCreationExpr.class)) {
            if (RISKY_TYPES.contains(o.getType().getNameAsString())) n++;
        }
        return n;
    }

    // ---------- compile + run helpers ----------

    /** Returns the directory on success, or null if the code does not compile. */
    static Path compileProgram(String code, Path dir) throws Exception {
        if (code == null || code.isBlank()) return null;
        Files.createDirectories(dir);
        Path javaFile = dir.resolve("Solution.java");
        Files.writeString(javaFile, code);

        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) throw new RuntimeException("No Java compiler found - run with a JDK, not a JRE.");

        ByteArrayOutputStream err = new ByteArrayOutputStream();
        int result = compiler.run(null, null, err, javaFile.toString());
        return result == 0 ? dir : null;
    }

    /** Runs the program with a hard 10 second timeout (output goes to a temp file so a hang cannot block us). */
        static String runProgram(Path dir, String input) throws Exception {
        return runProgram(dir, input, 10);
    }

    static String runProgram(Path dir, String input, int timeoutSec) throws Exception {
        String javaBin = Paths.get(System.getProperty("java.home"), "bin", "java").toString();
        File outFile = File.createTempFile("run_out", ".txt");
        try {
            ProcessBuilder pb = new ProcessBuilder(javaBin, "-cp", dir.toString(), "Solution");
            pb.redirectErrorStream(true);
            pb.redirectOutput(outFile);
            Process process = pb.start();

            try (OutputStream os = process.getOutputStream()) {
                os.write((input + "\n").getBytes());
                os.flush();
            } catch (IOException e) {
                // program may exit before reading all input - fine
            }

            if (!process.waitFor(timeoutSec, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                return "ERROR: timed out";
            }

            String out = Files.readString(outFile.toPath()).trim();
            if (process.exitValue() != 0) {
                String first = out.isEmpty() ? "exit code " + process.exitValue() : out.split("\n")[0];
                return "ERROR: " + first;
            }
            return out;
        } finally {
            outFile.delete();
        }
    }

    /** Deletes the temporary working folder and everything inside it. */
    static void deleteRecursively(Path dir) {
        try (var stream = Files.walk(dir)) {
            stream.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        } catch (IOException e) {
            // best effort cleanup
        }
    }

        // ---------- mutation testing ----------

    /** Makes up to MAX_MUTANTS copies of the program; each copy has exactly one operator changed (e.g. < becomes <=). */
    static List<String> generateMutants(String code) {
        List<Integer> candidates = new ArrayList<>();
        List<BinaryExpr> exprs = StaticJavaParser.parse(code).findAll(BinaryExpr.class);
        for (int i = 0; i < exprs.size(); i++) {
            if (MUTATIONS.containsKey(exprs.get(i).getOperator())) candidates.add(i);
        }

        List<String> mutants = new ArrayList<>();
        int count = Math.min(MAX_MUTANTS, candidates.size());
        for (int k = 0; k < count; k++) {
            int position = candidates.get(k * candidates.size() / count);   // spread picks across the program
            CompilationUnit copy = StaticJavaParser.parse(code);
            BinaryExpr target = copy.findAll(BinaryExpr.class).get(position);
            target.setOperator(MUTATIONS.get(target.getOperator()));
            mutants.add(copy.toString());
        }
        return mutants;
    }

    /**
     * Mutation score = killed mutants / valid mutants. A mutant is "killed" when at least one test input makes it
     * behave differently from the original program (different output, crash or timeout).
     * Mutants that do not compile are ignored. Returns NaN when no valid mutant could be built.
     */
    static double mutationScore(ImplResult r, List<String> tests, Path workDir) {
        List<String> mutants;
        try {
            mutants = generateMutants(r.code);
        } catch (Exception e) {
            return Double.NaN;
        }

        int valid = 0, killed = 0;
        for (int m = 0; m < mutants.size(); m++) {
            try {
                Path dir = compileProgram(mutants.get(m), workDir.resolve("impl" + r.index + "_mut" + m));
                if (dir == null) continue;                        // mutant does not compile - not counted
                valid++;
                for (int t = 0; t < tests.size() && t < r.normOutputs.size(); t++) {
                    String out = normalize(runProgram(dir, tests.get(t), MUTANT_TIMEOUT_SEC));
                    if (!out.equals(r.normOutputs.get(t))) {     // behaviour changed -> the tests caught it
                        killed++;
                        break;
                    }
                }
            } catch (Exception e) {
                // skip this mutant
            }
        }
        return valid == 0 ? Double.NaN : (double) killed / valid;
    }

    // ---------- output comparison ----------

    /** Makes outputs comparable: trims, collapses whitespace, "5.0" -> "5", ignores case. */
    static String normalize(String s) {
        if (s == null) return null;
        String t = s.trim().replaceAll("\\s+", " ");
        t = t.replaceAll("(?<=\\d)\\.0+(?=\\s|$)", "");
        return t.toLowerCase();
    }

    /** Returns the strict plurality output, or null on a tie / when nothing valid was produced. */
    static String findMajority(String[] outputs) {
        Map<String, Integer> counts = new HashMap<>();
        for (String o : outputs) {
            if (o != null && !o.startsWith("error:")) counts.merge(o, 1, Integer::sum);
        }
        String best = null;
        int bestCount = 0, second = 0;
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            if (e.getValue() > bestCount) {
                second = bestCount;
                bestCount = e.getValue();
                best = e.getKey();
            } else if (e.getValue() > second) {
                second = e.getValue();
            }
        }
        return (best != null && bestCount > second) ? best : null;
    }

    // ---------- console display helpers ----------

    /** Cuts long text so one huge test case cannot flood the terminal. */
    static String truncate(String s, int max) {
        return s.length() <= max ? s : s.substring(0, max) + "...";
    }

    /** Prints "n) first line" and then each further input line on its own real line, aligned underneath. */
    static void printNumberedInput(int number, String input) {
        String prefix = number + ") ";
        String pad = " ".repeat(prefix.length());
        String[] lines = input.split("\\R");
        System.out.println(prefix + "Input:    " + truncate(lines[0], MAX_LINE_CHARS));
        int shownLines = Math.min(lines.length, MAX_INPUT_LINES);
        for (int j = 1; j < shownLines; j++) {
            System.out.println(pad + "          " + truncate(lines[j], MAX_LINE_CHARS));
        }
        if (lines.length > shownLines) {
            System.out.println(pad + "          ... (" + (lines.length - shownLines) + " more lines)");
        }
    }

    /** Prints a titled, numbered list of test case inputs; multi-line inputs go on real new lines. */
    static void printTestCases(String title, List<String> tests) {
        System.out.println("\n" + title);
        for (int i = 0; i < tests.size(); i++) {
            printNumberedInput(i + 1, tests.get(i));
        }
    }

    // ---------- reporting ----------

    static String pct(double v) {
        return Double.isNaN(v) ? "-" : String.format("%.1f", v * 100);
    }

    static String[] toRow(ImplResult r) {
        return new String[]{
                "Prompt " + r.index + " code",
                r.model.replace("openai/", ""),
                r.compiled ? "Yes" : "No",
                pct(r.ast), pct(r.cfg), pct(r.dfg),
                    pct(r.testPass), pct(r.edgePassRate), pct(r.mutation),
                r.cyclomatic < 0 ? "-" : String.valueOf(r.cyclomatic),
                Double.isNaN(r.runtimeMs) ? "-" : String.format("%.0f", r.runtimeMs),
                r.security < 0 ? "-" : String.valueOf(r.security),
                pct(r.reliability)
        };
    }

    static String tableToString(String[] headers, List<String[]> rows) {
        int[] w = new int[headers.length];
        for (int i = 0; i < headers.length; i++) w[i] = headers[i].length();
        for (String[] row : rows) {
            for (int i = 0; i < row.length; i++) w[i] = Math.max(w[i], row[i].length());
        }
        StringBuilder sep = new StringBuilder("+");
        for (int width : w) sep.append("-".repeat(width + 2)).append("+");

        StringBuilder sb = new StringBuilder();
        sb.append(sep).append("\n");
        sb.append(rowToString(headers, w)).append("\n");
        sb.append(sep).append("\n");
        for (String[] row : rows) sb.append(rowToString(row, w)).append("\n");
        sb.append(sep);
        return sb.toString();
    }

    static String rowToString(String[] cells, int[] w) {
        StringBuilder sb = new StringBuilder("|");
        for (int i = 0; i < cells.length; i++) {
            sb.append(" ").append(String.format("%-" + w[i] + "s", cells[i])).append(" |");
        }
        return sb.toString();
    }

    static String interpretReliability(double score) {
        if (score >= 0.9) return "HIGH - implementations agree strongly, likely reliable";
        if (score >= 0.7) return "MODERATE - some inconsistency detected, review recommended";
        return "LOW - significant disagreement, likely contains a bug";
    }

    // ---------- Word document (.docx) writer, no external libraries needed ----------

    /** Saves ONLY the user's question and the results table, as a Word document with a real table. */
        static void saveResults(Path outFile, String label, String[] headers, List<String[]> rows,
                        List<ImplResult> impls) throws IOException {
        Path parent = outFile.getParent();
        if (parent != null) Files.createDirectories(parent);

        // landscape A4 usable width = 16838 - 2*720 twips; give Impl / Model a bit more room
        int cols = headers.length;
        int[] widths = new int[cols];
        for (int i = 0; i < cols; i++) widths[i] = (i == 0) ? 1500 : (i == 1 ? 1500 : 1120);

        StringBuilder body = new StringBuilder();
                body.append(docParagraph(label, true, 28));
        body.append(docParagraph("", false, 20));

        body.append("<w:tbl><w:tblPr>")
                .append("<w:tblW w:w=\"0\" w:type=\"auto\"/>")
                .append("<w:jc w:val=\"center\"/>")
                .append("<w:tblBorders>")
                .append(border("top")).append(border("left")).append(border("bottom"))
                .append(border("right")).append(border("insideH")).append(border("insideV"))
                .append("</w:tblBorders>")
                .append("<w:tblLayout w:type=\"fixed\"/>")
                .append("<w:tblCellMar><w:top w:w=\"60\" w:type=\"dxa\"/><w:left w:w=\"80\" w:type=\"dxa\"/>")
                .append("<w:bottom w:w=\"60\" w:type=\"dxa\"/><w:right w:w=\"80\" w:type=\"dxa\"/></w:tblCellMar>")
                .append("</w:tblPr><w:tblGrid>");
        for (int w : widths) body.append("<w:gridCol w:w=\"").append(w).append("\"/>");
        body.append("</w:tblGrid>");

        body.append(docRow(headers, widths, true));
        for (String[] row : rows) body.append(docRow(row, widths, false));
        body.append("</w:tbl>");
        body.append(docParagraph("", false, 20)); // Word needs a paragraph after a table
                // source code of every implementation, one after another, below the table
        for (ImplResult r : impls) {
                        String title = "Prompt " + r.index + " code"
                    + (r.compiled ? "" : " - did not compile");
            body.append(docHeading(title));
            if (r.code == null || r.code.isBlank()) {
                body.append(docCodeLine("No code was generated for this implementation."));
            } else {
                for (String line : r.code.split("\\R", -1)) body.append(docCodeLine(line));
            }
        }

        String documentXml =
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                        + "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>"
                        + body
                        + "<w:sectPr><w:pgSz w:w=\"16838\" w:h=\"11906\" w:orient=\"landscape\"/>"
                        + "<w:pgMar w:top=\"720\" w:right=\"720\" w:bottom=\"720\" w:left=\"720\" "
                        + "w:header=\"360\" w:footer=\"360\" w:gutter=\"0\"/></w:sectPr>"
                        + "</w:body></w:document>";

        String contentTypes =
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                        + "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
                        + "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
                        + "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
                        + "<Override PartName=\"/word/document.xml\" "
                        + "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
                        + "</Types>";

        String rels =
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                        + "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                        + "<Relationship Id=\"rId1\" "
                        + "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
                        + "Target=\"word/document.xml\"/></Relationships>";

        try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(outFile), StandardCharsets.UTF_8)) {
            addZipEntry(zip, "[Content_Types].xml", contentTypes);
            addZipEntry(zip, "_rels/.rels", rels);
            addZipEntry(zip, "word/document.xml", documentXml);
        }
    }

        /** Bold heading for one implementation's code section. */
    static String docHeading(String text) {
        return "<w:p><w:pPr><w:keepNext/><w:spacing w:before=\"240\" w:after=\"80\"/></w:pPr>"
                + docRun(text, true, 24) + "</w:p>";
    }

    /** One line of source code: monospaced font, light grey background, tight spacing. */
    static String docCodeLine(String line) {
        String text = line.replace("\t", "    ").replace("\r", "");
        return "<w:p><w:pPr><w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F2F2F2\"/>"
                + "<w:spacing w:before=\"0\" w:after=\"0\" w:line=\"240\" w:lineRule=\"auto\"/>"
                + "<w:rPr><w:sz w:val=\"18\"/></w:rPr></w:pPr>"
                + (text.isEmpty() ? "" :
                "<w:r><w:rPr><w:rFonts w:ascii=\"Consolas\" w:hAnsi=\"Consolas\" w:cs=\"Consolas\"/>"
                        + "<w:sz w:val=\"18\"/></w:rPr><w:t xml:space=\"preserve\">" + xmlEscape(text) + "</w:t></w:r>")
                + "</w:p>";
    }

    static void addZipEntry(ZipOutputStream zip, String name, String content) throws IOException {
        zip.putNextEntry(new ZipEntry(name));
        zip.write(content.getBytes(StandardCharsets.UTF_8));
        zip.closeEntry();
    }

    static String border(String side) {
        return "<w:" + side + " w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"444444\"/>";
    }

    static String docRun(String text, boolean bold, int halfPoints) {
        return "<w:r><w:rPr><w:rFonts w:ascii=\"Calibri\" w:hAnsi=\"Calibri\" w:cs=\"Calibri\"/>"
                + (bold ? "<w:b/>" : "")
                + "<w:sz w:val=\"" + halfPoints + "\"/></w:rPr>"
                + "<w:t xml:space=\"preserve\">" + xmlEscape(text) + "</w:t></w:r>";
    }

    static String docParagraph(String text, boolean bold, int halfPoints) {
        return "<w:p>" + (text.isEmpty() ? "" : docRun(text, bold, halfPoints)) + "</w:p>";
    }

    static String docRow(String[] cells, int[] widths, boolean header) {
        StringBuilder sb = new StringBuilder("<w:tr>");
        if (header) sb.append("<w:trPr><w:tblHeader/></w:trPr>");
        for (int i = 0; i < cells.length; i++) {
            sb.append("<w:tc><w:tcPr><w:tcW w:w=\"").append(widths[i]).append("\" w:type=\"dxa\"/>");
            if (header) sb.append("<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"D9E2F3\"/>");
            sb.append("<w:vAlign w:val=\"center\"/></w:tcPr>");
            sb.append("<w:p><w:pPr><w:jc w:val=\"center\"/></w:pPr>")
                    .append(docRun(cells[i], header, 18))
                    .append("</w:p></w:tc>");
        }
        return sb.append("</w:tr>").toString();
    }

    /** Escapes XML special characters and drops characters Word cannot store. */
    static String xmlEscape(String s) {
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == '&') sb.append("&amp;");
            else if (c == '<') sb.append("&lt;");
            else if (c == '>') sb.append("&gt;");
            else if (c == '"') sb.append("&quot;");
            else if (c == '\t' || c == '\n' || c == '\r' || c >= 0x20) sb.append(c);
        }
        return sb.toString();
    }

    // ---------- bug localization ----------

    static void localizeBugs(List<ImplResult> impls) {
        ImplResult trusted = null;
        for (ImplResult r : impls) {
            if (r.compiled && r.astProfile != null && (trusted == null || r.mismatches < trusted.mismatches)) {
                trusted = r;
            }
        }

        boolean any = false;
        System.out.println("\nBug Localization:");
        for (ImplResult r : impls) {
            if (!r.compiled) {
                System.out.println("SUSPECT: Implementation " + r.index + " did not compile.");
                any = true;
                continue;
            }
            if (r.mismatches == 0) continue;
            any = true;
            System.out.println("SUSPECT: Implementation " + r.index + " disagreed with the majority on "
                    + r.mismatches + " test case(s).");

            if (trusted != null && trusted != r && r.astProfile != null) {
                System.out.println("  Structural differences vs Implementation " + trusted.index + " (most trusted):");
                Set<String> keys = new HashSet<>(r.astProfile.keySet());
                keys.addAll(trusted.astProfile.keySet());
                for (String key : keys) {
                    int a = r.astProfile.getOrDefault(key, 0);
                    int b = trusted.astProfile.getOrDefault(key, 0);
                    if (a != b) {
                        System.out.println("    " + key + ": Impl " + r.index + "=" + a + ", Impl " + trusted.index + "=" + b);
                    }
                }
            }
        }
        if (!any) System.out.println("No disagreements detected - no implementation flagged as suspect.");
    }
}