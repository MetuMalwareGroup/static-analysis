def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_asm = tf.strings.regex_replace(lowercase, '<.*?>', ' ')
  stripped_asm = tf.strings.regex_replace(stripped_asm, '\n', ' ')
  stripped_asm = tf.strings.regex_replace(stripped_asm, '(bad)', ' ')
  stripped_asm = tf.strings.regex_replace(stripped_asm, 'align', ' ')
  stripped_asm = tf.strings.regex_replace(stripped_asm, 'int3', ' ')
  stripped_asm = tf.strings.regex_replace(stripped_asm, 'r"\([^()]*\)",', ' ')
  stripped_asm = tf.strings.regex_replace(stripped_asm, r'(0x[0-9a-fA-F]+)(?:)?', 'addr')
  stripped_asm = tf.strings.regex_replace(stripped_asm, 'r"([d|q]{0,}word |byte )"', "dqw")
  stripped_asm = tf.strings.regex_replace(stripped_asm, '[a-z]+mov|mov[a-z]*', 'mov')
  stripped_asm = tf.strings.regex_replace(stripped_asm, '[a-z]+mov|mov[a-z]*', 'mov')
  return tf.strings.regex_replace(stripped_asm,
                                  '[%s]' % re.escape(string.punctuation),
                                  ' ')
