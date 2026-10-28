#include "fm/chat_template.hpp"

#include "fm/common.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// A restricted Jinja2 interpreter for `tokenizer.chat_template`.
//
// Why an interpreter rather than llama.cpp's approach: llama.cpp keeps a table
// of C++ string builders and a chain of substring heuristics to pick one. That
// works for the ~60 templates it knows and silently produces the wrong prompt
// for everything else. The template in the file *is* the specification, so
// reading it is both smaller (one engine instead of sixty builders) and more
// correct. What it costs is a language subset, which is why an unsupported
// construct is a named, typed error rather than a best effort.
//
// The subset is what real templates use: for/if/set/namespace, expression
// operators, the handful of filters that appear in practice, whitespace
// control, and raise_exception. Anything else is refused.

namespace fm {
namespace {

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

struct Value;

using ValueList = std::vector<Value>;
// Field order is kept because `namespace()` printing and `dict` iteration are
// not used, but stable order makes debugging and error text deterministic.
struct Fields;

struct Value {
    enum class Type { Undefined, Null, Bool, Int, Double, String, List, Map, Namespace };

    Type type = Type::Undefined;
    bool boolean = false;
    int64_t integer = 0;
    double number = 0.0;
    std::string text;
    std::shared_ptr<ValueList> list;
    std::shared_ptr<Fields> fields;

    bool is_number() const { return type == Type::Int || type == Type::Double; }
    bool is_container() const { return type == Type::List || type == Type::Map ||
                                       type == Type::Namespace; }
    // Defined after `Fields`, because it dereferences it.
    bool truthy() const;
    double as_number() const {
        if (type == Type::Int) return static_cast<double>(integer);
        if (type == Type::Double) return number;
        if (type == Type::Bool) return boolean ? 1.0 : 0.0;
        return 0.0;
    }
};

struct Fields : std::vector<std::pair<std::string, Value>> {
    using std::vector<std::pair<std::string, Value>>::vector;
};

bool Value::truthy() const {
    switch (type) {
    case Type::Undefined:
    case Type::Null: return false;
    case Type::Bool: return boolean;
    case Type::Int: return integer != 0;
    case Type::Double: return number != 0.0;
    case Type::String: return !text.empty();
    case Type::List: return list && !list->empty();
    case Type::Map:
    case Type::Namespace: return fields && !fields->empty();
    }
    return false;
}

Value make_undefined() { return Value{}; }
Value make_null() {
    Value v;
    v.type = Value::Type::Null;
    return v;
}
Value make_bool(bool b) {
    Value v;
    v.type = Value::Type::Bool;
    v.boolean = b;
    return v;
}
Value make_int(int64_t i) {
    Value v;
    v.type = Value::Type::Int;
    v.integer = i;
    return v;
}
Value make_double(double d) {
    Value v;
    v.type = Value::Type::Double;
    v.number = d;
    return v;
}
Value make_string(std::string s) {
    Value v;
    v.type = Value::Type::String;
    v.text = std::move(s);
    return v;
}
Value make_list(ValueList items) {
    Value v;
    v.type = Value::Type::List;
    v.list = std::make_shared<ValueList>(std::move(items));
    return v;
}
Value make_map() {
    Value v;
    v.type = Value::Type::Map;
    v.fields = std::make_shared<Fields>();
    return v;
}
Value make_namespace() {
    Value v = make_map();
    v.type = Value::Type::Namespace;
    return v;
}

std::string format_double(double d) {
    if (std::isfinite(d) && d == static_cast<double>(static_cast<int64_t>(d)) &&
        std::fabs(d) < 1e15) {
        // Jinja prints 2.0, not 2, for a float.
        return std::to_string(static_cast<int64_t>(d)) + ".0";
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%g", d);
    return buf;
}

// Undefined renders as the empty string. Jinja raises for *arithmetic* on
// undefined, which is preserved below; rendering is the one place where the
// empty string is what every real template expects.
std::string to_string(const Value &v) {
    switch (v.type) {
    case Value::Type::Undefined:
    case Value::Type::Null: return "";
    case Value::Type::Bool: return v.boolean ? "true" : "false";
    case Value::Type::Int: return std::to_string(v.integer);
    case Value::Type::Double: return format_double(v.number);
    case Value::Type::String: return v.text;
    case Value::Type::List: {
        std::string out = "[";
        if (v.list) {
            for (size_t i = 0; i < v.list->size(); ++i) {
                if (i) out += ", ";
                const Value &item = (*v.list)[i];
                out += item.type == Value::Type::String ? "'" + item.text + "'" : to_string(item);
            }
        }
        return out + "]";
    }
    case Value::Type::Map:
    case Value::Type::Namespace: {
        std::string out = "{";
        if (v.fields) {
            for (size_t i = 0; i < v.fields->size(); ++i) {
                if (i) out += ", ";
                out += "'" + (*v.fields)[i].first + "': " + to_string((*v.fields)[i].second);
            }
        }
        return out + "}";
    }
    }
    return "";
}

const Value *field_of(const Value &v, const std::string &name) {
    if (!v.fields) return nullptr;
    for (const auto &entry : *v.fields)
        if (entry.first == name) return &entry.second;
    return nullptr;
}

// A distinct error type so `chat_template_supported` can tell "this template
// uses a feature I do not implement" from "this template is broken".
struct Unsupported : std::runtime_error {
    explicit Unsupported(const std::string &what) : std::runtime_error(what) {}
};

[[noreturn]] void unsupported(const std::string &what) {
    throw Unsupported("chat template uses an unsupported Jinja construct: " + what);
}

[[noreturn]] void bad_template(const std::string &what) {
    fail(ErrorCode::INVALID_ARGUMENT, "chat template: " + what);
}

// ---------------------------------------------------------------------------
// Chunk lexer
// ---------------------------------------------------------------------------

struct Chunk {
    enum class Kind { Text, Output, Statement };
    Kind kind = Kind::Text;
    std::string body;
    size_t offset = 0;
};

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

// Jinja's `-` markers strip *all* whitespace on the adjacent side, newlines
// included. Doing it here means the renderer never has to think about it.
void rstrip_in_place(std::string &s) {
    while (!s.empty() && is_space(s.back())) s.pop_back();
}
void lstrip_in_place(std::string &s) {
    size_t i = 0;
    while (i < s.size() && is_space(s[i])) ++i;
    s.erase(0, i);
}

std::vector<Chunk> lex_chunks(const std::string &src) {
    std::vector<Chunk> chunks;
    size_t pos = 0;
    bool strip_next = false;
    while (pos < src.size()) {
        size_t open = src.find_first_of("{", pos);
        if (open == std::string::npos) {
            std::string text = src.substr(pos);
            if (strip_next) lstrip_in_place(text);
            if (!text.empty()) chunks.push_back({Chunk::Kind::Text, text, pos});
            break;
        }
        if (open + 1 >= src.size()) {
            std::string text = src.substr(pos);
            if (strip_next) lstrip_in_place(text);
            if (!text.empty()) chunks.push_back({Chunk::Kind::Text, text, pos});
            break;
        }
        const char kind_ch = src[open + 1];
        Chunk::Kind kind;
        std::string closer;
        if (kind_ch == '{') {
            kind = Chunk::Kind::Output;
            closer = "}}";
        } else if (kind_ch == '%') {
            kind = Chunk::Kind::Statement;
            closer = "%}";
        } else if (kind_ch == '#') {
            // Comments render nothing; consume and continue.
            size_t end = src.find("#}", open + 2);
            if (end == std::string::npos) bad_template("unterminated comment");
            bool trim_after = false;
            if (end > open + 2 && src[end - 1] == '-') trim_after = true;
            std::string text = src.substr(pos, open - pos);
            if (strip_next) lstrip_in_place(text);
            if (trim_after) rstrip_in_place(text);
            if (!text.empty()) chunks.push_back({Chunk::Kind::Text, text, pos});
            strip_next = trim_after;
            pos = end + 2;
            continue;
        } else {
            // A lone '{' is just text.
            size_t next = open + 1;
            std::string text = src.substr(pos, next - pos);
            if (strip_next) lstrip_in_place(text);
            if (!text.empty()) chunks.push_back({Chunk::Kind::Text, text, pos});
            strip_next = false;
            pos = next;
            continue;
        }

        size_t body_start = open + 2;
        bool trim_before = false;
        if (body_start < src.size() && src[body_start] == '-') {
            trim_before = true;
            ++body_start;
        } else if (body_start < src.size() && src[body_start] == '+') {
            ++body_start; // explicit "do not strip"
        }
        size_t end = src.find(closer, body_start);
        if (end == std::string::npos) bad_template("unterminated " + std::string(kind_ch == '%' ? "{% %}" : "{{ }}"));
        bool trim_after = false;
        size_t body_end = end;
        if (body_end > body_start && src[body_end - 1] == '-') {
            trim_after = true;
            --body_end;
        } else if (body_end > body_start && src[body_end - 1] == '+') {
            --body_end;
        }
        std::string text = src.substr(pos, open - pos);
        if (strip_next) lstrip_in_place(text);
        if (trim_before) rstrip_in_place(text);
        if (!text.empty()) chunks.push_back({Chunk::Kind::Text, text, pos});

        Chunk chunk;
        chunk.kind = kind;
        chunk.body = src.substr(body_start, body_end - body_start);
        chunk.offset = open;
        chunks.push_back(chunk);
        strip_next = trim_after;
        pos = end + 2;
    }
    return chunks;
}

// ---------------------------------------------------------------------------
// Expression lexer
// ---------------------------------------------------------------------------

struct Token {
    enum class Kind { End, Name, Number, String, Op };
    Kind kind = Kind::End;
    std::string text;
    double number = 0.0;
    bool is_int = false;
    int64_t integer = 0;
    size_t offset = 0;
};

std::vector<Token> lex_expression(const std::string &src, size_t base_offset) {
    std::vector<Token> out;
    size_t i = 0;
    auto push_op = [&](const std::string &op) {
        Token t;
        t.kind = Token::Kind::Op;
        t.text = op;
        t.offset = base_offset + i;
        out.push_back(t);
        i += op.size();
    };
    while (i < src.size()) {
        const char c = src[i];
        if (is_space(c)) {
            ++i;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            size_t start = i;
            bool is_float = false;
            while (i < src.size() && (std::isdigit(static_cast<unsigned char>(src[i])) || src[i] == '.')) {
                if (src[i] == '.') is_float = true;
                ++i;
            }
            Token t;
            t.kind = Token::Kind::Number;
            t.text = src.substr(start, i - start);
            t.offset = base_offset + start;
            if (is_float) {
                t.number = std::strtod(t.text.c_str(), nullptr);
            } else {
                t.integer = std::strtoll(t.text.c_str(), nullptr, 10);
                t.is_int = true;
                t.number = static_cast<double>(t.integer);
            }
            out.push_back(t);
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = i;
            while (i < src.size() &&
                   (std::isalnum(static_cast<unsigned char>(src[i])) || src[i] == '_'))
                ++i;
            Token t;
            t.kind = Token::Kind::Name;
            t.text = src.substr(start, i - start);
            t.offset = base_offset + start;
            out.push_back(t);
            continue;
        }
        if (c == '\'' || c == '"') {
            const char quote = c;
            ++i;
            std::string value;
            bool closed = false;
            while (i < src.size()) {
                char ch = src[i++];
                if (ch == '\\' && i < src.size()) {
                    char esc = src[i++];
                    switch (esc) {
                    case 'n': value += '\n'; break;
                    case 't': value += '\t'; break;
                    case 'r': value += '\r'; break;
                    case '\\': value += '\\'; break;
                    case '\'': value += '\''; break;
                    case '"': value += '"'; break;
                    default: value += esc; break;
                    }
                    continue;
                }
                if (ch == quote) {
                    closed = true;
                    break;
                }
                value += ch;
            }
            if (!closed) bad_template("unterminated string literal");
            Token t;
            t.kind = Token::Kind::String;
            t.text = value;
            t.offset = base_offset + i;
            out.push_back(t);
            continue;
        }
        // Multi-character operators first.
        static const char *two_char[] = {"==", "!=", "<=", ">=", "//", "**"};
        bool matched = false;
        for (const char *op : two_char) {
            if (src.compare(i, 2, op) == 0) {
                push_op(op);
                matched = true;
                break;
            }
        }
        if (matched) continue;
        if (c == '<' || c == '>' || c == '+' || c == '-' || c == '*' || c == '/' || c == '%' ||
            c == '~' || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' ||
            c == ',' || c == ':' || c == '.' || c == '|' || c == '=') {
            push_op(std::string(1, c));
            continue;
        }
        bad_template("unexpected character '" + std::string(1, c) + "' in expression");
    }
    Token end;
    end.kind = Token::Kind::End;
    end.offset = base_offset + src.size();
    out.push_back(end);
    return out;
}

// ---------------------------------------------------------------------------
// Expression AST
// ---------------------------------------------------------------------------

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct Expr {
    enum class Kind {
        Literal,      // value
        Name,         // name
        Field,        // target.name
        Index,        // target[index]
        Slice,        // target[start:stop]
        Unary,        // op target
        Binary,       // left op right
        Logical,      // left (and|or) right
        Compare,      // left op right
        Concat,       // left ~ right
        Call,         // callee(args...)
        Filter,       // target | name(args)
        Test,         // target is [not] name(args)
        ListLit,      // [a, b]
    };
    Kind kind = Kind::Literal;
    Value value;
    std::string name;
    std::string op;
    ExprPtr left, right, target;
    std::vector<ExprPtr> args;

    // For calls and filters, keyword arguments (namespace(a=1), default(x, true)).
    std::vector<std::pair<std::string, ExprPtr>> kwargs;
    // For slices.
    bool has_start = false, has_stop = false;
};

ExprPtr make_literal(Value v) {
    auto e = std::make_shared<Expr>();
    e->kind = Expr::Kind::Literal;
    e->value = std::move(v);
    return e;
}

class ExpressionParser {
public:
    ExpressionParser(std::vector<Token> tokens, const std::string &source)
        : tokens_(std::move(tokens)), source_(source) {}

    ExprPtr parse_all() {
        ExprPtr e = parse_expression();
        if (peek().kind != Token::Kind::End) {
            unsupported("trailing input in expression near '" + peek().text + "'");
        }
        return e;
    }

    // Parses only a name, for `{% set x = ... %}` targets and loop variables.
    std::string parse_name_only() {
        if (peek().kind != Token::Kind::Name) bad_template("expected a name");
        std::string n = next().text;
        if (peek().kind != Token::Kind::End) unsupported("complex assignment target");
        return n;
    }

private:
    std::vector<Token> tokens_;
    const std::string &source_;
    size_t pos_ = 0;

    const Token &peek(size_t ahead = 0) const {
        size_t i = pos_ + ahead;
        return i < tokens_.size() ? tokens_[i] : tokens_.back();
    }
    const Token &next() { return tokens_[std::min(pos_++, tokens_.size() - 1)]; }
    bool accept_op(const char *op) {
        if (peek().kind == Token::Kind::Op && peek().text == op) {
            ++pos_;
            return true;
        }
        return false;
    }
    bool accept_name(const char *name) {
        if (peek().kind == Token::Kind::Name && peek().text == name) {
            ++pos_;
            return true;
        }
        return false;
    }
    void expect_op(const char *op) {
        if (!accept_op(op)) bad_template(std::string("expected '") + op + "' in expression");
    }

    ExprPtr parse_expression() { return parse_or(); }

    ExprPtr parse_or() {
        ExprPtr left = parse_and();
        while (peek().kind == Token::Kind::Name && peek().text == "or") {
            next();
            auto e = std::make_shared<Expr>();
            e->kind = Expr::Kind::Logical;
            e->op = "or";
            e->left = left;
            e->right = parse_and();
            left = e;
        }
        return left;
    }

    ExprPtr parse_and() {
        ExprPtr left = parse_not();
        while (peek().kind == Token::Kind::Name && peek().text == "and") {
            next();
            auto e = std::make_shared<Expr>();
            e->kind = Expr::Kind::Logical;
            e->op = "and";
            e->left = left;
            e->right = parse_not();
            left = e;
        }
        return left;
    }

    ExprPtr parse_not() {
        if (peek().kind == Token::Kind::Name && peek().text == "not") {
            next();
            auto e = std::make_shared<Expr>();
            e->kind = Expr::Kind::Unary;
            e->op = "not";
            e->target = parse_not();
            return e;
        }
        return parse_comparison();
    }

    ExprPtr parse_comparison() {
        ExprPtr left = parse_concat();
        while (true) {
            std::string op;
            if (peek().kind == Token::Kind::Op &&
                (peek().text == "==" || peek().text == "!=" || peek().text == "<" ||
                 peek().text == "<=" || peek().text == ">" || peek().text == ">=")) {
                op = next().text;
            } else if (peek().kind == Token::Kind::Name && peek().text == "in") {
                next();
                op = "in";
            } else if (peek().kind == Token::Kind::Name && peek().text == "not" &&
                       peek(1).kind == Token::Kind::Name && peek(1).text == "in") {
                next();
                next();
                op = "not in";
            } else if (peek().kind == Token::Kind::Name && peek().text == "is") {
                next();
                bool negated = accept_name("not");
                if (peek().kind != Token::Kind::Name) bad_template("expected a test name after 'is'");
                std::string test_name = next().text;
                auto e = std::make_shared<Expr>();
                e->kind = Expr::Kind::Test;
                e->name = test_name;
                e->op = negated ? "not" : "";
                e->target = left;
                if (accept_op("(")) {
                    if (!accept_op(")")) {
                        e->args.push_back(parse_expression());
                        while (accept_op(",")) e->args.push_back(parse_expression());
                        expect_op(")");
                    }
                }
                left = e;
                continue;
            } else {
                break;
            }
            auto e = std::make_shared<Expr>();
            e->kind = Expr::Kind::Compare;
            e->op = op;
            e->left = left;
            e->right = parse_concat();
            left = e;
        }
        return left;
    }

    ExprPtr parse_concat() {
        ExprPtr left = parse_additive();
        while (accept_op("~")) {
            auto e = std::make_shared<Expr>();
            e->kind = Expr::Kind::Concat;
            e->left = left;
            e->right = parse_additive();
            left = e;
        }
        return left;
    }

    ExprPtr parse_additive() {
        ExprPtr left = parse_multiplicative();
        while (peek().kind == Token::Kind::Op && (peek().text == "+" || peek().text == "-")) {
            std::string op = next().text;
            auto e = std::make_shared<Expr>();
            e->kind = Expr::Kind::Binary;
            e->op = op;
            e->left = left;
            e->right = parse_multiplicative();
            left = e;
        }
        return left;
    }

    ExprPtr parse_multiplicative() {
        ExprPtr left = parse_unary();
        while (peek().kind == Token::Kind::Op &&
               (peek().text == "*" || peek().text == "/" || peek().text == "//" ||
                peek().text == "%")) {
            std::string op = next().text;
            auto e = std::make_shared<Expr>();
            e->kind = Expr::Kind::Binary;
            e->op = op;
            e->left = left;
            e->right = parse_unary();
            left = e;
        }
        return left;
    }

    ExprPtr parse_unary() {
        if (peek().kind == Token::Kind::Op && (peek().text == "-" || peek().text == "+")) {
            std::string op = next().text;
            auto e = std::make_shared<Expr>();
            e->kind = Expr::Kind::Unary;
            e->op = op;
            e->target = parse_unary();
            return e;
        }
        return parse_filters();
    }

    // Filters bind tighter than arithmetic in Jinja: `a + b|trim + c` applies
    // the filter to `b` alone, which is exactly how the llama3 template relies
    // on `message['content'] | trim` sitting between two `+`.
    ExprPtr parse_filters() {
        ExprPtr target = parse_postfix();
        while (accept_op("|")) {
            if (peek().kind != Token::Kind::Name) bad_template("expected a filter name");
            auto e = std::make_shared<Expr>();
            e->kind = Expr::Kind::Filter;
            e->name = next().text;
            e->target = target;
            if (accept_op("(")) {
                if (!accept_op(")")) {
                    do {
                        if (peek().kind == Token::Kind::Name && peek(1).kind == Token::Kind::Op &&
                            peek(1).text == "=") {
                            std::string kw = next().text;
                            next(); // '='
                            e->kwargs.emplace_back(kw, parse_expression());
                        } else {
                            e->args.push_back(parse_expression());
                        }
                    } while (accept_op(","));
                    expect_op(")");
                }
            }
            target = e;
        }
        return target;
    }

    ExprPtr parse_postfix() {
        ExprPtr target = parse_primary();
        while (true) {
            if (accept_op(".")) {
                if (peek().kind != Token::Kind::Name) bad_template("expected an attribute name after '.'");
                auto e = std::make_shared<Expr>();
                e->kind = Expr::Kind::Field;
                e->name = next().text;
                e->target = target;
                target = e;
            } else if (accept_op("[")) {
                bool has_start = false, has_stop = false;
                ExprPtr start_expr, stop_expr;
                if (!(peek().kind == Token::Kind::Op && peek().text == ":")) {
                    start_expr = parse_expression();
                    has_start = true;
                }
                if (accept_op(":")) {
                    has_stop = true;
                    if (!(peek().kind == Token::Kind::Op && peek().text == "]")) {
                        stop_expr = parse_expression();
                    }
                    expect_op("]");
                    auto e = std::make_shared<Expr>();
                    e->kind = Expr::Kind::Slice;
                    e->target = target;
                    e->left = start_expr;
                    e->right = stop_expr;
                    e->has_start = has_start;
                    e->has_stop = has_stop && stop_expr != nullptr;
                    target = e;
                    continue;
                }
                expect_op("]");
                auto e = std::make_shared<Expr>();
                e->kind = Expr::Kind::Index;
                e->target = target;
                e->left = start_expr;
                target = e;
            } else if (accept_op("(")) {
                auto e = std::make_shared<Expr>();
                e->kind = Expr::Kind::Call;
                e->target = target;
                if (!accept_op(")")) {
                    do {
                        if (peek().kind == Token::Kind::Name && peek(1).kind == Token::Kind::Op &&
                            peek(1).text == "=") {
                            std::string kw = next().text;
                            next();
                            e->kwargs.emplace_back(kw, parse_expression());
                        } else {
                            e->args.push_back(parse_expression());
                        }
                    } while (accept_op(","));
                    expect_op(")");
                }
                target = e;
            } else {
                break;
            }
        }
        return target;
    }

    ExprPtr parse_primary() {
        const Token &t = peek();
        switch (t.kind) {
        case Token::Kind::Number: {
            next();
            return make_literal(t.is_int ? make_int(t.integer) : make_double(t.number));
        }
        case Token::Kind::String:
            next();
            return make_literal(make_string(t.text));
        case Token::Kind::Name: {
            std::string name = t.text;
            next();
            if (name == "true" || name == "True") return make_literal(make_bool(true));
            if (name == "false" || name == "False") return make_literal(make_bool(false));
            if (name == "none" || name == "None" || name == "null") return make_literal(make_null());
            auto e = std::make_shared<Expr>();
            e->kind = Expr::Kind::Name;
            e->name = name;
            return e;
        }
        case Token::Kind::Op:
            if (t.text == "(") {
                next();
                ExprPtr inner = parse_expression();
                expect_op(")");
                return inner;
            }
            if (t.text == "[") {
                next();
                auto e = std::make_shared<Expr>();
                e->kind = Expr::Kind::ListLit;
                if (!accept_op("]")) {
                    do {
                        e->args.push_back(parse_expression());
                    } while (accept_op(","));
                    expect_op("]");
                }
                return e;
            }
            break;
        case Token::Kind::End:
            break;
        }
        bad_template("unexpected end of expression");
    }
};

// ---------------------------------------------------------------------------
// Statement AST
// ---------------------------------------------------------------------------

struct Node;
using NodePtr = std::shared_ptr<Node>;

struct Node {
    enum class Kind { Text, Output, If, For, Set };
    Kind kind = Kind::Text;
    std::string text;                     // Text
    ExprPtr expr;                         // Output, Set value
    std::vector<std::pair<ExprPtr, std::vector<NodePtr>>> branches;  // If
    std::vector<NodePtr> otherwise;       // If/For else body
    std::vector<NodePtr> body;            // For body
    std::string var;                      // For variable / Set target
    bool field_set = false;               // Set target was `ns.field`
    ExprPtr iterable;                     // For
    ExprPtr condition;                    // For filter
};

class TemplateParser {
public:
    explicit TemplateParser(std::vector<Chunk> chunks) : chunks_(std::move(chunks)) {}

    std::vector<NodePtr> parse() {
        std::vector<NodePtr> body = parse_body({});
        if (pos_ < chunks_.size()) bad_template("unexpected '" + first_word(chunks_[pos_].body) + "'");
        return body;
    }

private:
    std::vector<Chunk> chunks_;
    size_t pos_ = 0;

    static std::string first_word(const std::string &s) {
        size_t i = 0;
        while (i < s.size() && is_space(s[i])) ++i;
        size_t j = i;
        while (j < s.size() && !is_space(s[j])) ++j;
        return s.substr(i, j - i);
    }

    // Parses until one of `terminators` is the next statement keyword. The
    // terminating chunk is left unconsumed so the caller can inspect it.
    std::vector<NodePtr> parse_body(const std::vector<std::string> &terminators) {
        std::vector<NodePtr> body;
        while (pos_ < chunks_.size()) {
            const Chunk &chunk = chunks_[pos_];
            if (chunk.kind == Chunk::Kind::Text) {
                auto n = std::make_shared<Node>();
                n->kind = Node::Kind::Text;
                n->text = chunk.body;
                body.push_back(n);
                ++pos_;
                continue;
            }
            if (chunk.kind == Chunk::Kind::Output) {
                auto n = std::make_shared<Node>();
                n->kind = Node::Kind::Output;
                n->expr = ExpressionParser(lex_expression(chunk.body, chunk.offset), chunk.body)
                              .parse_all();
                body.push_back(n);
                ++pos_;
                continue;
            }
            const std::string keyword = first_word(chunk.body);
            for (const auto &term : terminators)
                if (keyword == term) return body;

            if (keyword == "if") {
                body.push_back(parse_if(chunk));
            } else if (keyword == "for") {
                body.push_back(parse_for(chunk));
            } else if (keyword == "set") {
                body.push_back(parse_set(chunk));
                ++pos_;
            } else if (keyword == "raise_exception") {
                // Statement form of the same function.
                auto n = std::make_shared<Node>();
                n->kind = Node::Kind::Output;
                n->expr = ExpressionParser(lex_expression(chunk.body, chunk.offset), chunk.body)
                              .parse_all();
                body.push_back(n);
                ++pos_;
            } else if (keyword == "endif" || keyword == "endfor" || keyword == "else" ||
                       keyword == "elif") {
                bad_template("unexpected '" + keyword + "'");
            } else {
                unsupported("{% " + keyword + " %}");
            }
        }
        if (!terminators.empty()) {
            std::string expected;
            for (const auto &t : terminators) expected += (expected.empty() ? "" : " or ") + t;
            bad_template("missing " + expected);
        }
        return body;
    }

    NodePtr parse_if(const Chunk &chunk) {
        auto n = std::make_shared<Node>();
        n->kind = Node::Kind::If;
        std::string rest = chunk.body.substr(chunk.body.find("if") + 2);
        ExprPtr cond = ExpressionParser(lex_expression(rest, chunk.offset), rest).parse_all();
        ++pos_;
        while (true) {
            std::vector<NodePtr> branch_body = parse_body({"elif", "else", "endif"});
            n->branches.emplace_back(cond, std::move(branch_body));
            if (pos_ >= chunks_.size()) bad_template("missing endif");
            const std::string keyword = first_word(chunks_[pos_].body);
            if (keyword == "elif") {
                const std::string expr_text =
                    chunks_[pos_].body.substr(chunks_[pos_].body.find("elif") + 4);
                cond = ExpressionParser(lex_expression(expr_text, chunks_[pos_].offset), expr_text)
                           .parse_all();
                ++pos_;
                continue;
            }
            if (keyword == "else") {
                ++pos_;
                n->otherwise = parse_body({"endif"});
                if (pos_ >= chunks_.size()) bad_template("missing endif");
                ++pos_;
                break;
            }
            ++pos_; // endif
            break;
        }
        return n;
    }

    NodePtr parse_for(const Chunk &chunk) {
        auto n = std::make_shared<Node>();
        n->kind = Node::Kind::For;
        std::string rest = chunk.body.substr(chunk.body.find("for") + 3);
        // `<var> in <iterable> [if <condition>]`, remembering that the filter
        // itself may contain the word "in".
        size_t in_pos = rest.find(" in ");
        if (in_pos == std::string::npos) bad_template("malformed for statement");
        n->var = trim(rest.substr(0, in_pos));
        std::string iter_text = rest.substr(in_pos + 4);
        size_t if_pos = std::string::npos;
        // Find a trailing ` if ` that is not inside brackets.
        int depth = 0;
        for (size_t i = 0; i + 4 <= iter_text.size(); ++i) {
            const char c = iter_text[i];
            if (c == '(' || c == '[' || c == '{') ++depth;
            else if (c == ')' || c == ']' || c == '}') --depth;
            else if (depth == 0 && iter_text.compare(i, 4, " if ") == 0) if_pos = i;
        }
        if (if_pos != std::string::npos) {
            n->condition = ExpressionParser(lex_expression(iter_text.substr(if_pos + 4), chunk.offset),
                                            iter_text)
                               .parse_all();
            iter_text = iter_text.substr(0, if_pos);
        }
        n->iterable = ExpressionParser(lex_expression(iter_text, chunk.offset), iter_text).parse_all();
        ++pos_;
        n->body = parse_body({"endfor", "else"});
        if (pos_ >= chunks_.size()) bad_template("missing endfor");
        if (first_word(chunks_[pos_].body) == "else") {
            ++pos_;
            n->otherwise = parse_body({"endfor"});
            if (pos_ >= chunks_.size()) bad_template("missing endfor");
        }
        ++pos_; // endfor
        return n;
    }

    NodePtr parse_set(const Chunk &chunk) {
        auto n = std::make_shared<Node>();
        n->kind = Node::Kind::Set;
        std::string rest = trim(chunk.body.substr(chunk.body.find("set") + 3));
        size_t eq = rest.find('=');
        if (eq == std::string::npos) bad_template("malformed set statement");
        std::string target = trim(rest.substr(0, eq));
        std::string value_text = rest.substr(eq + 1);
        size_t dot = target.find('.');
        if (dot != std::string::npos) {
            // `set ns.attr = ...` mutates a namespace created by namespace().
            n->var = trim(target.substr(0, dot));
            n->text = trim(target.substr(dot + 1));
            n->field_set = true;
        } else {
            n->var = target;
        }
        n->expr = ExpressionParser(lex_expression(value_text, chunk.offset), value_text).parse_all();
        return n;
    }
};

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

struct Scope {
    std::vector<std::pair<std::string, Value>> vars;

    Value *find(const std::string &name) {
        for (auto &entry : vars)
            if (entry.first == name) return &entry.second;
        return nullptr;
    }
};

class Evaluator {
public:
    explicit Evaluator(std::vector<Scope> &scopes) : scopes_(scopes) {}

    Value eval(const ExprPtr &e) {
        switch (e->kind) {
        case Expr::Kind::Literal: return e->value;
        case Expr::Kind::Name: return lookup(e->name);
        case Expr::Kind::Field: return field(eval(e->target), e->name);
        case Expr::Kind::Index: return index(eval(e->target), eval(e->left));
        case Expr::Kind::Slice: return slice(eval(e->target), e->left, e->right, e->has_start,
                                             e->has_stop);
        case Expr::Kind::Unary: {
            Value v = eval(e->target);
            if (e->op == "not") return make_bool(!v.truthy());
            if (e->op == "-") return numeric_negate(v);
            return v;
        }
        case Expr::Kind::Binary: return binary(e->op, eval(e->left), eval(e->right));
        case Expr::Kind::Logical: {
            Value left = eval(e->left);
            if (e->op == "and") return left.truthy() ? eval(e->right) : left;
            return left.truthy() ? left : eval(e->right);
        }
        case Expr::Kind::Compare: return compare(e->op, eval(e->left), eval(e->right));
        case Expr::Kind::Concat: return make_string(to_string(eval(e->left)) + to_string(eval(e->right)));
        case Expr::Kind::Call: return call(e);
        case Expr::Kind::Filter: return filter(e);
        case Expr::Kind::Test: return test(e);
        case Expr::Kind::ListLit: {
            ValueList items;
            for (const auto &arg : e->args) items.push_back(eval(arg));
            return make_list(std::move(items));
        }
        }
        return make_undefined();
    }

    Value lookup(const std::string &name) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            if (Value *v = it->find(name)) return *v;
        }
        return make_undefined();
    }

    void assign(const std::string &name, const Value &value) {
        // Assign to the nearest enclosing scope that already declares the name,
        // otherwise declare it here. This is what makes `{% set content = ... %}`
        // inside a for loop visible on the next iteration.
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            if (Value *v = it->find(name)) {
                *v = value;
                return;
            }
        }
        scopes_.back().vars.emplace_back(name, value);
    }

private:
    std::vector<Scope> &scopes_;

    static Value numeric_negate(const Value &v) {
        if (!v.is_number()) return v;
        return v.type == Value::Type::Int ? make_int(-v.integer) : make_double(-v.number);
    }

    Value field(const Value &target, const std::string &name) {
        if (const Value *found = field_of(target, name)) return *found;
        if (target.type == Value::Type::List && name == "length") {
            return make_int(target.list ? static_cast<int64_t>(target.list->size()) : 0);
        }
        return make_undefined();
    }

    Value index(const Value &target, const Value &key) {
        if (target.type == Value::Type::List && key.is_number()) {
            int64_t i = static_cast<int64_t>(key.as_number());
            if (!target.list || i < 0 || i >= static_cast<int64_t>(target.list->size()))
                return make_undefined();
            return (*target.list)[static_cast<size_t>(i)];
        }
        if (target.is_container() && key.type == Value::Type::String) return field(target, key.text);
        if (target.type == Value::Type::String && key.is_number()) {
            int64_t i = static_cast<int64_t>(key.as_number());
            if (i < 0) i += static_cast<int64_t>(target.text.size());
            if (i < 0 || i >= static_cast<int64_t>(target.text.size())) return make_undefined();
            return make_string(std::string(1, target.text[static_cast<size_t>(i)]));
        }
        return make_undefined();
    }

    Value slice(const Value &target, const ExprPtr &start, const ExprPtr &stop, bool has_start,
                bool has_stop) {
        int64_t n = 0;
        if (target.type == Value::Type::List && target.list)
            n = static_cast<int64_t>(target.list->size());
        else if (target.type == Value::Type::String)
            n = static_cast<int64_t>(target.text.size());
        else
            return make_undefined();
        int64_t from = has_start ? static_cast<int64_t>(eval(start).as_number()) : 0;
        int64_t to = has_stop ? static_cast<int64_t>(eval(stop).as_number()) : n;
        if (from < 0) from += n;
        if (to < 0) to += n;
        from = std::max<int64_t>(0, std::min(from, n));
        to = std::max<int64_t>(from, std::min(to, n));
        if (target.type == Value::Type::String)
            return make_string(target.text.substr(static_cast<size_t>(from),
                                                  static_cast<size_t>(to - from)));
        ValueList items(target.list->begin() + from, target.list->begin() + to);
        return make_list(std::move(items));
    }

    Value binary(const std::string &op, const Value &l, const Value &r) {
        if ((op == "+") && l.type == Value::Type::String && r.type == Value::Type::String)
            return make_string(l.text + r.text);
        if (op == "+" && l.type == Value::Type::List && r.type == Value::Type::List) {
            ValueList items = *l.list;
            items.insert(items.end(), r.list->begin(), r.list->end());
            return make_list(std::move(items));
        }
        if (!l.is_number() || !r.is_number()) {
            // Undefined arithmetic is a hard error in Jinja, and silently
            // treating it as zero hides a template bug behind wrong output.
            bad_template("arithmetic on a non-number ('" + op + "')");
        }
        const bool ints = l.type == Value::Type::Int && r.type == Value::Type::Int;
        const double a = l.as_number(), b = r.as_number();
        if (op == "+") return ints ? make_int(l.integer + r.integer) : make_double(a + b);
        if (op == "-") return ints ? make_int(l.integer - r.integer) : make_double(a - b);
        if (op == "*") return ints ? make_int(l.integer * r.integer) : make_double(a * b);
        if (op == "/") {
            if (b == 0.0) bad_template("division by zero");
            return make_double(a / b);
        }
        if (op == "//") {
            if (b == 0.0) bad_template("division by zero");
            return make_int(static_cast<int64_t>(std::floor(a / b)));
        }
        if (op == "%") {
            if (b == 0.0) bad_template("modulo by zero");
            return make_int(static_cast<int64_t>(a) % static_cast<int64_t>(b));
        }
        bad_template("unknown operator '" + op + "'");
    }

    bool equal(const Value &l, const Value &r) {
        if (l.is_number() && r.is_number()) return l.as_number() == r.as_number();
        if (l.type == Value::Type::Bool || r.type == Value::Type::Bool)
            return l.truthy() == r.truthy();
        if (l.type == Value::Type::String && r.type == Value::Type::String) return l.text == r.text;
        if (l.type == Value::Type::Undefined && r.type == Value::Type::Null) return true;
        if (l.type == Value::Type::Null && r.type == Value::Type::Undefined) return true;
        if (l.type != r.type) return false;
        if (l.type == Value::Type::String) return l.text == r.text;
        return to_string(l) == to_string(r);
    }

    Value compare(const std::string &op, const Value &l, const Value &r) {
        if (op == "==") return make_bool(equal(l, r));
        if (op == "!=") return make_bool(!equal(l, r));
        if (op == "in" || op == "not in") {
            bool found = false;
            if (r.type == Value::Type::List && r.list) {
                for (const auto &item : *r.list)
                    if (equal(item, l)) {
                        found = true;
                        break;
                    }
            } else if (r.type == Value::Type::String) {
                found = r.text.find(to_string(l)) != std::string::npos;
            } else if (r.is_container() && l.type == Value::Type::String) {
                found = field_of(r, l.text) != nullptr;
            }
            return make_bool(op == "in" ? found : !found);
        }
        if (!l.is_number() || !r.is_number())
            bad_template("ordering comparison on a non-number ('" + op + "')");
        const double a = l.as_number(), b = r.as_number();
        if (op == "<") return make_bool(a < b);
        if (op == "<=") return make_bool(a <= b);
        if (op == ">") return make_bool(a > b);
        if (op == ">=") return make_bool(a >= b);
        bad_template("unknown comparison '" + op + "'");
    }

    Value test(const ExprPtr &e) {
        const Value v = eval(e->target);
        bool result = false;
        if (e->name == "defined") result = v.type != Value::Type::Undefined;
        else if (e->name == "undefined") result = v.type == Value::Type::Undefined;
        else if (e->name == "none") result = v.type == Value::Type::Null;
        else if (e->name == "string") result = v.type == Value::Type::String;
        else if (e->name == "number") result = v.is_number();
        else if (e->name == "boolean") result = v.type == Value::Type::Bool;
        else if (e->name == "mapping" || e->name == "iterable" || e->name == "sequence")
            result = v.is_container() || v.type == Value::Type::String;
        else unsupported("test 'is " + e->name + "'");
        if (e->op == "not") result = !result;
        return make_bool(result);
    }

    Value call(const ExprPtr &e) {
        if (e->target->kind != Expr::Kind::Name) {
            // Method call on a value: `value.strip()`, `list.join(',')`.
            const Value receiver = eval(e->target->target);
            return method(receiver, e->target->name, e);
        }
        const std::string &name = e->target->name;
        std::vector<Value> args;
        for (const auto &arg : e->args) args.push_back(eval(arg));
        auto kwarg = [&](const std::string &key) -> Value {
            for (const auto &entry : e->kwargs)
                if (entry.first == key) return eval(entry.second);
            return make_undefined();
        };
        if (name == "raise_exception") {
            fail(ErrorCode::INVALID_ARGUMENT,
                 "chat template refused this conversation: " +
                     (args.empty() ? std::string("raise_exception") : to_string(args[0])));
        }
        if (name == "namespace") {
            Value ns = make_namespace();
            for (const auto &entry : e->kwargs) {
                const Value v = eval(entry.second);
                ns.fields->emplace_back(entry.first, v);
            }
            (void)kwarg;
            return ns;
        }
        if (name == "range") {
            const int64_t n = args.empty() ? 0 : static_cast<int64_t>(args[0].as_number());
            ValueList items;
            for (int64_t i = 0; i < n; ++i) items.push_back(make_int(i));
            return make_list(std::move(items));
        }
        if (name == "str" || name == "string") return make_string(args.empty() ? "" : to_string(args[0]));
        if (name == "int") return make_int(args.empty() ? 0 : static_cast<int64_t>(args[0].as_number()));
        if (name == "dict") {
            Value m = make_map();
            for (const auto &entry : e->kwargs) m.fields->emplace_back(entry.first, eval(entry.second));
            return m;
        }
        unsupported("function '" + name + "()'");
    }

    Value method(const Value &receiver, const std::string &name, const ExprPtr &e) {
        std::vector<Value> args;
        for (const auto &arg : e->args) args.push_back(eval(arg));
        if (receiver.type != Value::Type::String) unsupported("method '." + name + "()'");
        const std::string &s = receiver.text;
        if (name == "strip") return make_string(trim(s));
        if (name == "lstrip") {
            std::string out = s;
            lstrip_in_place(out);
            return make_string(out);
        }
        if (name == "rstrip") {
            std::string out = s;
            rstrip_in_place(out);
            return make_string(out);
        }
        if (name == "upper") {
            std::string out = s;
            for (char &c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            return make_string(out);
        }
        if (name == "lower") {
            std::string out = s;
            for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return make_string(out);
        }
        if (name == "replace") {
            if (args.size() < 2) bad_template("replace() needs two arguments");
            std::string out = s;
            const std::string from = to_string(args[0]), to = to_string(args[1]);
            if (from.empty()) return make_string(out);
            size_t at = 0;
            while ((at = out.find(from, at)) != std::string::npos) {
                out.replace(at, from.size(), to);
                at += to.size();
            }
            return make_string(out);
        }
        if (name == "startswith") return make_bool(!args.empty() && s.rfind(to_string(args[0]), 0) == 0);
        if (name == "endswith") {
            const std::string suffix = args.empty() ? "" : to_string(args[0]);
            return make_bool(s.size() >= suffix.size() &&
                             s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0);
        }
        if (name == "join") {
            if (args.empty() || args[0].type != Value::Type::List) return make_string("");
            std::string out;
            for (size_t i = 0; i < args[0].list->size(); ++i) {
                if (i) out += s;
                out += to_string((*args[0].list)[i]);
            }
            return make_string(out);
        }
        if (name == "split") {
            const std::string sep = args.empty() ? " " : to_string(args[0]);
            ValueList items;
            if (sep.empty()) {
                for (char c : s) items.push_back(make_string(std::string(1, c)));
            } else {
                size_t at = 0, found;
                while ((found = s.find(sep, at)) != std::string::npos) {
                    items.push_back(make_string(s.substr(at, found - at)));
                    at = found + sep.size();
                }
                items.push_back(make_string(s.substr(at)));
            }
            return make_list(std::move(items));
        }
        unsupported("string method '." + name + "()'");
    }

    Value filter(const ExprPtr &e) {
        std::vector<Value> args;
        for (const auto &arg : e->args) args.push_back(eval(arg));
        auto kwarg = [&](const std::string &key, const Value &fallback) -> Value {
            for (const auto &entry : e->kwargs)
                if (entry.first == key) return eval(entry.second);
            return fallback;
        };
        const Value v = eval(e->target);
        const std::string &name = e->name;
        if (name == "trim" || name == "strip") return make_string(trim(to_string(v)));
        if (name == "length" || name == "count") {
            if (v.type == Value::Type::List) return make_int(static_cast<int64_t>(v.list->size()));
            if (v.type == Value::Type::String) return make_int(static_cast<int64_t>(v.text.size()));
            if (v.fields) return make_int(static_cast<int64_t>(v.fields->size()));
            return make_int(0);
        }
        if (name == "upper" || name == "lower") {
            std::string s = to_string(v);
            for (char &c : s)
                c = static_cast<char>(name == "upper" ? std::toupper(static_cast<unsigned char>(c))
                                                      : std::tolower(static_cast<unsigned char>(c)));
            return make_string(s);
        }
        if (name == "string") return make_string(to_string(v));
        if (name == "int") return make_int(static_cast<int64_t>(v.as_number()));
        if (name == "float") return make_double(v.as_number());
        if (name == "first") {
            if (v.type == Value::Type::List && v.list && !v.list->empty()) return v.list->front();
            if (v.type == Value::Type::String && !v.text.empty()) return make_string(v.text.substr(0, 1));
            return make_undefined();
        }
        if (name == "last") {
            if (v.type == Value::Type::List && v.list && !v.list->empty()) return v.list->back();
            if (v.type == Value::Type::String && !v.text.empty())
                return make_string(v.text.substr(v.text.size() - 1));
            return make_undefined();
        }
        if (name == "list") {
            if (v.type == Value::Type::List) return v;
            ValueList items;
            if (v.type == Value::Type::String)
                for (char c : v.text) items.push_back(make_string(std::string(1, c)));
            return make_list(std::move(items));
        }
        if (name == "join") {
            const std::string sep = args.empty() ? "" : to_string(args[0]);
            if (v.type != Value::Type::List || !v.list) return make_string("");
            std::string out;
            for (size_t i = 0; i < v.list->size(); ++i) {
                if (i) out += sep;
                out += to_string((*v.list)[i]);
            }
            return make_string(out);
        }
        if (name == "replace") {
            if (args.size() < 2) bad_template("replace filter needs two arguments");
            std::string out = to_string(v);
            const std::string from = to_string(args[0]), to = to_string(args[1]);
            if (from.empty()) return make_string(out);
            size_t at = 0;
            while ((at = out.find(from, at)) != std::string::npos) {
                out.replace(at, from.size(), to);
                at += to.size();
            }
            return make_string(out);
        }
        if (name == "default" || name == "d") {
            const Value fallback = args.empty() ? make_string("") : args[0];
            const bool boolean_default = kwarg("boolean", make_bool(false)).truthy();
            if (v.type == Value::Type::Undefined) return fallback;
            if (boolean_default && !v.truthy()) return fallback;
            return v;
        }
        if (name == "tojson") {
            // Only used to embed strings in a few templates; JSON-escape.
            std::string out = "\"";
            for (char c : to_string(v)) {
                if (c == '"' || c == '\\') out += '\\';
                if (c == '\n') {
                    out += "\\n";
                    continue;
                }
                out += c;
            }
            return make_string(out + "\"");
        }
        unsupported("filter '| " + name + "'");
    }
};

class Renderer {
public:
    Renderer(std::vector<Scope> &scopes) : scopes_(scopes) {}

    void render(const std::vector<NodePtr> &body, std::string &out) {
        for (const NodePtr &node : body) render_node(*node, out);
    }

private:
    std::vector<Scope> &scopes_;

    void render_node(const Node &node, std::string &out) {
        switch (node.kind) {
        case Node::Kind::Text: out += node.text; break;
        case Node::Kind::Output: {
            Evaluator ev(scopes_);
            out += to_string(ev.eval(node.expr));
            break;
        }
        case Node::Kind::Set: {
            Evaluator ev(scopes_);
            const Value v = ev.eval(node.expr);
            if (!node.field_set) {
                ev.assign(node.var, v);
            } else {
                Value target = ev.lookup(node.var);
                if (target.type != Value::Type::Namespace || !target.fields)
                    bad_template("set on a field of '" + node.var + "', which is not a namespace");
                bool replaced = false;
                for (auto &entry : *target.fields) {
                    if (entry.first == node.text) {
                        entry.second = v;
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) target.fields->emplace_back(node.text, v);
            }
            break;
        }
        case Node::Kind::If: {
            for (const auto &branch : node.branches) {
                Evaluator ev(scopes_);
                if (ev.eval(branch.first).truthy()) {
                    render(branch.second, out);
                    return;
                }
            }
            render(node.otherwise, out);
            break;
        }
        case Node::Kind::For: {
            Evaluator ev(scopes_);
            const Value iterable = ev.eval(node.iterable);
            ValueList items;
            if (iterable.type == Value::Type::List && iterable.list) {
                // A `for ... if ...` filter needs the loop variable bound, so
                // it is applied inside the iteration, not here.
                for (const Value &item : *iterable.list) items.push_back(item);
            } else if (iterable.type == Value::Type::String) {
                for (char c : iterable.text) items.push_back(make_string(std::string(1, c)));
            } else if (iterable.is_container() && iterable.fields) {
                for (const auto &entry : *iterable.fields)
                    items.push_back(make_string(entry.first));
            } else if (iterable.type != Value::Type::Undefined) {
                bad_template("cannot iterate over a scalar");
            }

            scopes_.emplace_back();
            size_t emitted = 0;
            const size_t total = items.size();
            for (size_t i = 0; i < total; ++i) {
                Scope &loop_scope = scopes_.back();
                auto bind = [&](const std::string &name, const Value &v) {
                    bool replaced = false;
                    for (auto &entry : loop_scope.vars) {
                        if (entry.first == name) {
                            entry.second = v;
                            replaced = true;
                            break;
                        }
                    }
                    if (!replaced) loop_scope.vars.emplace_back(name, v);
                };
                bind(node.var, items[i]);
                Value loop = make_map();
                loop.fields->emplace_back("index0", make_int(static_cast<int64_t>(i)));
                loop.fields->emplace_back("index", make_int(static_cast<int64_t>(i) + 1));
                loop.fields->emplace_back("first", make_bool(i == 0));
                loop.fields->emplace_back("last", make_bool(i + 1 == total));
                loop.fields->emplace_back("length", make_int(static_cast<int64_t>(total)));
                loop.fields->emplace_back("revindex", make_int(static_cast<int64_t>(total - i)));
                loop.fields->emplace_back("revindex0", make_int(static_cast<int64_t>(total - i) - 1));
                bind("loop", loop);
                if (node.condition) {
                    Evaluator cond(scopes_);
                    if (!cond.eval(node.condition).truthy()) continue;
                }
                render(node.body, out);
                ++emitted;
            }
            scopes_.pop_back();
            if (emitted == 0 && !node.otherwise.empty()) render(node.otherwise, out);
            break;
        }
        }
    }
};

// ---------------------------------------------------------------------------
// Built-in templates
// ---------------------------------------------------------------------------

// Kept as Jinja text, not as C++ string builders, so a built-in is rendered by
// exactly the same interpreter as a checkpoint's own template and cannot drift
// from it.
//
// Only families whose markers are unambiguous are listed. Everything else is
// resolved from token *evidence* (see `detect_from_tokens`), because guessing a
// template from an architecture name produces a plausible-looking prompt that
// is silently wrong.
const char *TEMPLATE_CHATML =
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

const char *TEMPLATE_LLAMA3 =
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + "
    "message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";

const char *TEMPLATE_LLAMA2 =
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{% set loop_messages = messages %}{% set system_message = false %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if loop.index0 == 0 and system_message != false %}"
    "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
    "{% else %}{% set content = message['content'] %}{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + content | trim + ' ' + eos_token }}"
    "{% endif %}"
    "{% endfor %}";

const char *TEMPLATE_GEMMA =
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if (message['role'] == 'user') %}"
    "{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}"
    "{% elif (message['role'] == 'assistant') %}"
    "{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}";

const char *TEMPLATE_PHI3 =
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}";

const char *TEMPLATE_DEEPSEEK2 =
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n\\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}"
    "{% elif message['role'] == 'system' %}{{ message['content'] + '\\n\\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}";

const char *TEMPLATE_RAW =
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'user' %}{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] }}"
    "{% endif %}"
    "{% endfor %}";

// OrionStarAI/Orion-14B-Chat, as llama.cpp's LLM_CHAT_TEMPLATE_ORION emits it:
// the system turn is folded into the first human turn, and each human turn ends
// with an assistant marker so the model continues from there.
const char *TEMPLATE_ORION =
    "{% set ns = namespace(system='') %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{% set ns.system = ns.system + message['content'] %}"
    "{% elif message['role'] == 'user' %}"
    "{{ 'Human: ' }}{% if ns.system != '' %}{{ ns.system + '\\n\\n' }}{% set ns.system = '' %}{% endif %}"
    "{{ message['content'] + '\\n\\nAssistant: ' + eos_token }}"
    "{% else %}{{ message['content'] + eos_token }}{% endif %}"
    "{% endfor %}";

const char *TEMPLATE_ZEPHYR =
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + eos_token + '\\n' }}"
    "{% elif message['role'] == 'system' %}{{ '<|system|>\\n' + message['content'] + eos_token + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + eos_token + '\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}";

const char *TEMPLATE_VICUNA =
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + '\\n\\n' + bos_token }}"
    "{% for message in messages[1:] %}"
    "{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + eos_token + '\\n' }}"
    "{% endif %}{% endfor %}"
    "{% else %}"
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + eos_token + '\\n' }}"
    "{% endif %}{% endfor %}"
    "{% endif %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}";

struct BuiltinTemplate {
    const char *name;
    const char *text;
};

const BuiltinTemplate *builtin_by_name(const std::string &name) {
    static const BuiltinTemplate table[] = {
        {"chatml", TEMPLATE_CHATML}, {"llama3", TEMPLATE_LLAMA3}, {"llama2", TEMPLATE_LLAMA2},
        {"gemma", TEMPLATE_GEMMA},   {"phi3", TEMPLATE_PHI3},     {"deepseek2", TEMPLATE_DEEPSEEK2},
        {"zephyr", TEMPLATE_ZEPHYR}, {"vicuna", TEMPLATE_VICUNA},   {"raw", TEMPLATE_RAW},
        {"orion", TEMPLATE_ORION},
    };
    for (const auto &entry : table)
        if (name == entry.name) return &entry;
    return nullptr;
}

} // namespace

std::string render_chat_template(const std::string &template_text,
                                 const std::vector<ChatMessage> &messages,
                                 const ChatTemplateConfig &config) {
    if (messages.empty()) fail(ErrorCode::INVALID_ARGUMENT, "a chat needs at least one message");

    std::vector<NodePtr> body;
    try {
        std::vector<Chunk> chunks = lex_chunks(template_text);
        TemplateParser parser(std::move(chunks));
        body = parser.parse();
    } catch (const Unsupported &e) {
        fail(ErrorCode::NOT_IMPLEMENTED, e.what());
    }

    // `messages` is a list of maps, which is what every template indexes into.
    ValueList message_values;
    for (const ChatMessage &m : messages) {
        Value entry = make_map();
        entry.fields->emplace_back("role", make_string(m.role));
        entry.fields->emplace_back("content", make_string(m.content));
        message_values.push_back(std::move(entry));
    }

    std::vector<Scope> scopes(1);
    scopes.back().vars.emplace_back("messages", make_list(std::move(message_values)));
    scopes.back().vars.emplace_back("add_generation_prompt", make_bool(config.add_generation_prompt));
    scopes.back().vars.emplace_back("bos_token", make_string(config.bos_token));
    scopes.back().vars.emplace_back("eos_token", make_string(config.eos_token));
    // Templates that reference tool/vision support see these as undefined,
    // which is falsy, and therefore take their no-tools path.
    std::string out;
    try {
        Renderer renderer(scopes);
        renderer.render(body, out);
    } catch (const Unsupported &e) {
        // The internal signal becomes a typed engine error so callers can tell
        // "this checkpoint needs a Jinja feature I do not have" from a bug.
        fail(ErrorCode::NOT_IMPLEMENTED, e.what());
    }
    return out;
}

std::string builtin_chat_template_name(const std::string &architecture) {
    // Only families whose markers are unambiguous are mapped by name. The
    // loader additionally resolves from token evidence, which is stronger.
    static const std::map<std::string, const char *> by_arch = {
        {"qwen2", "chatml"},     {"qwen3", "chatml"},   {"qwen2moe", "chatml"},
        {"qwen3moe", "chatml"},  {"glm4", "chatml"},    {"deepseek2", "deepseek2"},
        {"phi3", "phi3"},        {"gemma", "gemma"},    {"gemma2", "gemma"},
        {"gemma3", "gemma"},     {"llama", "llama2"},   {"smollm", "llama2"},
        {"smollm2", "llama2"},   {"tinyllama", "llama2"}, {"minicpm", "chatml"},
        {"phi2", "chatml"},      {"internlm2", "chatml"}, {"exaone", "chatml"},
        {"stablelm", "chatml"},  {"orion", "orion"},
    };
    auto it = by_arch.find(architecture);
    return it == by_arch.end() ? std::string() : std::string(it->second);
}

std::string builtin_chat_template_by_name(const std::string &name) {
    const BuiltinTemplate *entry = builtin_by_name(name);
    return entry ? std::string(entry->text) : std::string();
}

std::string builtin_chat_template(const std::string &architecture) {
    return builtin_chat_template_by_name(builtin_chat_template_name(architecture));
}

std::string builtin_chat_template_from_tokens(
    const std::string &architecture, bool has_llama3_header, bool has_im_start,
    bool has_start_of_turn, bool has_phi3_markers, bool has_inst, bool has_zephyr_markers) {
    // Order matters: a vocabulary can carry more than one marker set (Qwen and
    // Phi-3 both ship `<|im_start|>`-free vocabularies, but a merged tokenizer
    // could carry both), so the most specific marker wins.
    if (has_llama3_header) return "llama3";
    if (has_start_of_turn) return "gemma";
    if (has_phi3_markers) return "phi3";
    if (has_zephyr_markers) return "zephyr";
    if (has_im_start) return "chatml";
    if (has_inst) return "llama2";
    return builtin_chat_template_name(architecture);
}

std::string resolve_chat_template(const std::string &checkpoint_template,
                                  const std::string &architecture,
                                  const std::string &override_text) {
    if (!override_text.empty()) {
        // A bare template name is a built-in; anything else is taken literally.
        if (const BuiltinTemplate *entry = builtin_by_name(override_text))
            return std::string(entry->text);
        return override_text;
    }
    if (!checkpoint_template.empty()) return checkpoint_template;
    return builtin_chat_template(architecture);
}

bool chat_template_supported(const std::string &template_text, std::string *reason) {
    if (template_text.empty()) {
        if (reason) *reason = "no template";
        return false;
    }
    std::vector<ChatMessage> probe = {
        {"user", "ping"},
        {"assistant", "pong"},
        {"user", "ping"},
    };
    ChatTemplateConfig config;
    config.bos_token = "<s>";
    config.eos_token = "</s>";
    try {
        (void)render_chat_template(template_text, probe, config);
        return true;
    } catch (const Unsupported &e) {
        if (reason) *reason = e.what();
        return false;
    } catch (const Error &e) {
        if (e.code == ErrorCode::NOT_IMPLEMENTED) {
            if (reason) *reason = e.what();
            return false;
        }
        // The template parsed and ran and simply refused this conversation
        // (gemma rejects a system role, llama2 rejects non-alternating turns).
        // That is a working template, not an unsupported one.
        if (reason) *reason = std::string("renders, refused the probe conversation: ") + e.what();
        return true;
    }
}

} // namespace fm
