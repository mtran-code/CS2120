def count_vowels():
    """
    This function prompts the user for a string input.
    It then counts the number of each vowel (a, e, i, o, u) in the given string.
    This may include sentences or words. Numbers and consonants will be ignored.
    Vowels will be counted regardless of being uppercase or lowercase.
    The function then prints the number of each vowel in the format:
    [input] contains [x] a's, [x] e's, [x] i's, [x] o's, and [x] u's.
    """
    
    words = str(input("What is your input? "))

    a_count = 0
    e_count = 0
    i_count = 0
    o_count = 0
    u_count = 0

    for i in range(len(words)):
        if words[i] == "a" or words[i] == "A":
            a_count += 1
        elif words[i] == "e" or words[i] == "E":
            e_count += 1
        elif words[i] == "i" or words[i] == "I":
            i_count += 1
        elif words[i] == "o" or words[i] == "O":
            o_count += 1
        elif words[i] == "u" or words[i] == "U":
            u_count += 1
        else:
            continue
    print(
        '"' + words + '"' + " contains " + str(a_count) + " a's, " + str(e_count) + " e's, "
        + str(i_count) + " i's, " + str(o_count) + " o's, and " + str(u_count) + " u's.")


count_vowels()

# Test Outputs:
# Input 1:      Hello, how are you doing?
# Output 1:     "Hello, how are you doing?" contains 1 a's, 2 e's, 1 i's, 4 o's, and 1 u's.

# Input 2:      AAAAH I LOVE SCREAMING!! I'LL SCREAM MY HEART OUT!
# Output 2:     "AAAAH I LOVE SCREAMING!! I'LL SCREAM MY HEART OUT!" contains 7 a's, 4 e's, 3 i's, 2 o's, and 1 u's.

# Input 3:      Oh boy, I sure hope I pass this Lecture Activity.
# Output 3:     "Oh boy, I sure hope I pass this Lecture Activity." contains 2 a's, 4 e's, 5 i's, 3 o's, and 2 u's.
