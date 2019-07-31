test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> len(df_train)
          75
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> len(df_test)
          75
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> df_test.iloc[74,1]
          3.0
          """,
          'hidden': False,
          'locked': False
        }
      ],
      'scored': True,
      'setup': '',
      'teardown': '',
      'type': 'doctest'
    }
  ]
}
