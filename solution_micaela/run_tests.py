import importlib.util
import os
import textwrap

MOD_PATH = os.path.join(os.path.dirname(__file__), 'query_agent.py')

spec = importlib.util.spec_from_file_location('qa', MOD_PATH)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)

tests = [
    "¿Cuál es el balance de V-12345678?",
    "¿Cómo abro una cuenta en BANCO HENRY?",
    "¿Cómo solicito una tarjeta de crédito?",
    "¿Cómo realizo una transferencia bancaria?",
    "¿Qué es una transferencia bancaria?"
]

def run():
    for q in tests:
        print('\n' + '='*80)
        print('Query:', q)
        try:
            # Retrieved docs if available
            retrieved = None
            if hasattr(qa, 'retrieve_docs'):
                try:
                    retrieved = qa.retrieve_docs(q, top_k=4)
                except Exception as e:
                    retrieved = f'Error retrieving docs: {e}'

            # Route and respond
            try:
                resp = qa.route_and_respond(q)
            except Exception as e:
                resp = f'Error in route_and_respond: {e}'

            print('\nResponse:\n')
            print(textwrap.indent(str(resp), '  '))

            print('\nRetrieved fragments:\n')
            if isinstance(retrieved, list):
                for i, d in enumerate(retrieved):
                    print(f'  --- fragment {i+1} (source={d.get("meta", {}).get("source")})')
                    print(textwrap.indent(d.get('text', '')[:1000], '    '))
                    print()
            else:
                print('  ', retrieved)

        except Exception as e:
            print('Error running test:', e)

if __name__ == '__main__':
    run()
